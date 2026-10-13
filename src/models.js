// The models and precisions offered in the UI.
// Every file below was checked to exist in the model repo's onnx/ folder on the Hugging Face Hub.

export const MODELS = [
  // Same backbone, different input resolution: "small" looks at a 512x512 image instead of
  // 384x384 (1.8x more pixels), so it is slower but better at small, distant objects.
  { id: 'onnx-community/rfdetr_nano-ONNX', label: 'RF-DETR nano · 384 px' },
  { id: 'onnx-community/rfdetr_small-ONNX', label: 'RF-DETR small · 512 px' },
];

// Precision = how many bits are used to store each of the model's numbers (its "weights").
//   fp32  32-bit floating point. The original model; the reference everything is compared to.
//   fp16  16-bit floating point. Half the size. Needs a GPU that can do 16-bit float math.
//   q8    "Quantized": weights rounded to 8-bit integers (256 possible values) plus a scale
//         factor to map them back to real numbers. ~4x smaller than fp32.
//   q4    Same idea with 4-bit integers (16 possible values) for the biggest layers.
//   q4f16 4-bit weights, and the rest of the math done in 16-bit floats.
// The repo also ships int8 / uint8 / bnb4 variants; uint8 is byte-for-byte the same file as q8,
// and int8 / bnb4 behaved like q8 / q4 in our tests, so they are left out to keep the menu short.
export const PRECISIONS = [
  { dtype: 'fp32', label: 'fp32 · full precision', file: 'model.onnx' },
  { dtype: 'fp16', label: 'fp16 · half precision', file: 'model_fp16.onnx', needsF16: true },
  { dtype: 'q8', label: 'q8 · 8-bit weights', file: 'model_quantized.onnx' },
  { dtype: 'q4', label: 'q4 · 4-bit weights', file: 'model_q4.onnx' },
  { dtype: 'q4f16', label: 'q4f16 · 4-bit + fp16', file: 'model_q4f16.onnx', needsF16: true },
];

export function findModel(id) {
  return MODELS.find((m) => m.id === id);
}

export function findPrecision(dtype) {
  return PRECISIONS.find((p) => p.dtype === dtype);
}

// Where Transformers.js downloads a given model file from (also its key in the browser cache).
export function modelFileUrl(modelId, file) {
  return `https://huggingface.co/${modelId}/resolve/main/onnx/${file}`;
}
