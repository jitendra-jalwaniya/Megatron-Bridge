# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron Bridge for the text decoder of Gemma 4 "Unified" checkpoints (e.g. 12B).

``google/gemma-4-12B-it`` ships as ``Gemma4UnifiedForConditionalGeneration``
(``model_type="gemma4_unified"``): a new vision embedder and an audio tower around a
text decoder that is architecturally the Gemma 4 dense decoder (dual RoPE,
sliding/global attention, ``attention_k_eq_v`` on global layers, no PLE, no shared
KV). The decoder weights use the same ``model.language_model.*`` names as
``Gemma4ForConditionalGeneration``, so this bridge reuses the Gemma 4 VL bridge in
its ``text`` conversion mode: the HF checkpoint is converted to a text-only
``Gemma4DenseModelConfig``/``GPTModel`` and the vision/audio weights are ignored.

Usage::

  AutoBridge.from_hf_pretrained("google/gemma-4-12B-it").get_model_config()
    └─ Gemma4UnifiedTextBridge (registered for Gemma4UnifiedForConditionalGeneration)
         ├─ get_model_config()  → Gemma4DenseModelConfig (text decoder only)
         └─ mapping_registry()  → Gemma4Bridge._dense_mapping_registry(),
                                  HF prefix ``model.language_model.``
"""

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge


@MegatronModelBridge.register_bridge(
    source="Gemma4UnifiedForConditionalGeneration",
    target=GPTModel,
    provider=Gemma4DenseProvider,
    model_type="gemma4_unified",
)
class Gemma4UnifiedTextBridge(Gemma4VLBridge):
    """Text-only bridge for Gemma 4 Unified checkpoints.

    The multimodal towers of the Unified family (``vision_embedder``, ``audio_tower``)
    differ from Gemma 4 VL and are not modelled in Megatron yet, so conversion is
    always text-only regardless of ``GEMMA4_CONVERSION_MODE``.
    """

    def _conversion_mode(self) -> str:
        return "text"


__all__ = ["Gemma4UnifiedTextBridge"]
