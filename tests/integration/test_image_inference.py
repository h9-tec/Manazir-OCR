import pytest
from docustruct.model import InferenceManager, BatchInputItem


def test_inference_image(simple_text_image):
    try:
        manager = InferenceManager(method="hf")
    except (OSError, Exception) as e:
        # Skip test if model can't be loaded (e.g., private repo, network issues)
        pytest.skip(f"Could not load model: {e}")
    
    batch = [
        BatchInputItem(
            image=simple_text_image,
            prompt_type="ocr_layout",
        )
    ]
    # Use more tokens to allow complete generation
    outputs = manager.generate(batch, max_output_tokens=512)
    assert len(outputs) == 1
    output = outputs[0]
    
    # Check that output was generated successfully
    assert not output.error, "Model generation should not have errors"
    assert len(output.markdown) > 10, f"Expected meaningful output, got: {output.markdown[:50]}"
    
    # Check for text content (could be in markdown, HTML, or raw format)
    # The model might format it differently, so check multiple formats
    output_text = (output.markdown + " " + output.html + " " + output.raw).lower()
    assert "hello" in output_text or "world" in output_text, (
        f"Expected 'Hello' or 'World' in output. "
        f"Markdown: {output.markdown[:100]}, "
        f"HTML: {output.html[:100]}, "
        f"Raw: {output.raw[:100]}"
    )

    # Check that chunks were parsed
    chunks = output.chunks
    assert len(chunks) > 0, "Expected at least one chunk in output"
