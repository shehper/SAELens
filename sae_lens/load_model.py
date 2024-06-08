from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule


def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:  # pragma: no cover
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    elif model_class_name == "nanogpt":
        model = load_nanogpt_model(ckpt_path=model_name, 
                                   device=device)
        return model
    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {model_class_name}")

def load_nanogpt_model(ckpt_path, device):
    # load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"""Error loading checkpoint: {e}, 
                Expects full path to the model checkpoint as model_name.""")

    # get config
    from transformer_lens import HookedTransformerConfig
    model_config = checkpoint['model_args']
    cfg = HookedTransformerConfig(
        n_layers=model_config["n_layer"],
        d_model=model_config["n_embd"],
        d_head=int(model_config["n_embd"]/ model_config["n_head"]),
        n_heads=model_config["n_head"],
        d_mlp=model_config["n_embd"] * 4,
        d_vocab=model_config["vocab_size"],
        n_ctx=model_config["block_size"],
        act_fn="gelu",
        normalization_type="LN",
        )

    # load state dict
    from transformer_lens.loading_from_pretrained import convert_nanogpt_weights
    state_dict = checkpoint['model']
    new_state_dict = convert_nanogpt_weights(old_state_dict=state_dict, cfg=cfg)
    model = HookedTransformer(cfg)
    model.load_state_dict(new_state_dict, strict=False)

    return model