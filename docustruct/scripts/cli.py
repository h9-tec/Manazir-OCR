import json
from pathlib import Path
from typing import List, Optional

import click

from docustruct.input import load_file, load_pdf_images, load_image
from docustruct.model import InferenceManager
from docustruct.model.schema import BatchInputItem
from docustruct.model import (
    create_model,
    create_recommended_model,
    list_models,
    MODEL_REGISTRY,
    ModelTier,
)


def get_supported_files(input_path: Path) -> List[Path]:
    """Get list of supported image/PDF files from path."""
    supported_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".tiff",
        ".bmp",
    }

    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            return [input_path]
        else:
            raise click.BadParameter(f"Unsupported file type: {input_path.suffix}")

    elif input_path.is_dir():
        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        # Deduplicate while preserving order
        seen = set()
        unique_files = []
        for p in files:
            if p not in seen:
                seen.add(p)
                unique_files.append(p)
        return sorted(unique_files)

    else:
        raise click.BadParameter(f"Path does not exist: {input_path}")


def save_merged_output(
    output_dir: Path,
    file_name: str,
    results: List,
    save_images: bool = True,
    save_html: bool = True,
    paginate_output: bool = False,
):
    """Save merged OCR results for all pages to output directory."""
    # Create subfolder for this file
    safe_name = Path(file_name).stem
    file_output_dir = output_dir / safe_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Merge all pages
    all_markdown = []
    all_html = []
    all_metadata = []
    total_tokens = 0
    total_chunks = 0
    total_images = 0

    # Process each page result
    for page_num, result in enumerate(results):
        # Add page separator for multi-page documents
        if page_num > 0 and paginate_output:
            all_markdown.append(f"\n\n{page_num}" + "-" * 48 + "\n\n")
            all_html.append(f"\n\n<!-- Page {page_num + 1} -->\n\n")

        all_markdown.append(result.markdown)
        all_html.append(result.html)

        # Accumulate metadata
        total_tokens += result.token_count
        total_chunks += len(result.chunks)
        total_images += len(result.images)

        page_metadata = {
            "page_num": page_num,
            "page_box": result.page_box,
            "token_count": result.token_count,
            "num_chunks": len(result.chunks),
            "num_images": len(result.images),
        }
        all_metadata.append(page_metadata)

        # Save extracted images if requested
        if save_images and result.images:
            images_dir = file_output_dir
            images_dir.mkdir(exist_ok=True)

            for img_name, pil_image in result.images.items():
                img_path = images_dir / img_name
                pil_image.save(img_path)

    # Save merged markdown
    markdown_path = file_output_dir / f"{safe_name}.md"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("".join(all_markdown))

    # Save merged HTML if requested
    if save_html:
        html_path = file_output_dir / f"{safe_name}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("".join(all_html))

    # Save combined metadata
    metadata = {
        "file_name": file_name,
        "num_pages": len(results),
        "total_token_count": total_tokens,
        "total_chunks": total_chunks,
        "total_images": total_images,
        "pages": all_metadata,
    }
    metadata_path = file_output_dir / f"{safe_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"  Saved: {markdown_path} ({len(results)} page(s))")


@click.group()
@click.version_option(version="0.2.0", prog_name="Manazir OCR")
def cli():
    """Manazir OCR - Arabic optics-inspired Multi-Model OCR Framework"""
    pass


@cli.command(name="process")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--method",
    type=click.Choice(["hf", "vllm"], case_sensitive=False),
    default="vllm",
    help="Inference method: 'hf' for local model, 'vllm' for vLLM server.",
)
@click.option(
    "--page-range",
    type=str,
    default=None,
    help="Page range for PDFs (e.g., '1-5,7,9-12'). Only applicable to PDF files.",
)
@click.option(
    "--max-output-tokens",
    type=int,
    default=None,
    help="Maximum number of output tokens per page.",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help="Maximum number of parallel workers for vLLM inference.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Maximum number of retries for vLLM inference.",
)
@click.option(
    "--include-images/--no-images",
    default=True,
    help="Include images in output.",
)
@click.option(
    "--include-headers-footers/--no-headers-footers",
    default=False,
    help="Include page headers and footers in output.",
)
@click.option(
    "--save-html/--no-html",
    default=True,
    help="Save HTML output files.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of pages to process in a batch.",
)
@click.option(
    "--paginate_output",
    is_flag=True,
    default=False,
)
def process(
    input_path: Path,
    output_path: Path,
    method: str,
    page_range: str,
    max_output_tokens: int,
    max_workers: int,
    max_retries: int,
    include_images: bool,
    include_headers_footers: bool,
    save_html: bool,
    batch_size: int,
    paginate_output: bool,
):
    if method == "hf":
        click.echo(
            "When using '--method hf', ensure that the batch size is set correctly.  We will default to batch size of 1."
        )
        if batch_size is None:
            batch_size = 1
    elif method == "vllm":
        if batch_size is None:
            batch_size = 28

    click.echo("Manazir OCR CLI - Starting OCR processing")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Method: {method}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    click.echo(f"\nLoading model with method '{method}'...")
    model = InferenceManager(method=method)
    click.echo("Model loaded successfully.")

    # Get files to process
    files_to_process = get_supported_files(input_path)
    click.echo(f"\nFound {len(files_to_process)} file(s) to process.")

    if not files_to_process:
        click.echo("No supported files found. Exiting.")
        return

    # Process each file
    for file_idx, file_path in enumerate(files_to_process, 1):
        click.echo(
            f"\n[{file_idx}/{len(files_to_process)}] Processing: {file_path.name}"
        )

        try:
            # Load images from file
            config = {"page_range": page_range} if page_range else {}
            images = load_file(str(file_path), config)
            click.echo(f"  Loaded {len(images)} page(s)")

            # Accumulate all results for this document
            all_results = []

            # Process pages in batches
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch_images = images[batch_start:batch_end]

                # Create batch input items
                batch = [
                    BatchInputItem(image=img, prompt_type="ocr_layout")
                    for img in batch_images
                ]

                # Run inference
                click.echo(f"  Processing pages {batch_start + 1}-{batch_end}...")

                # Build kwargs for generate
                generate_kwargs = {
                    "include_images": include_images,
                    "include_headers_footers": include_headers_footers,
                }

                if max_output_tokens is not None:
                    generate_kwargs["max_output_tokens"] = max_output_tokens

                if method == "vllm":
                    if max_workers is not None:
                        generate_kwargs["max_workers"] = max_workers
                    if max_retries is not None:
                        generate_kwargs["max_retries"] = max_retries

                results = model.generate(batch, **generate_kwargs)
                all_results.extend(results)

            # Save merged output for all pages
            save_merged_output(
                output_path,
                file_path.name,
                all_results,
                save_images=include_images,
                save_html=save_html,
                paginate_output=paginate_output,
            )

            click.echo(f"  Completed: {file_path.name}")

        except Exception as e:
            click.echo(f"  Error processing {file_path.name}: {e}", err=True)
            continue

    click.echo(f"\nProcessing complete. Results saved to: {output_path}")


@cli.command(name="ocr")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "output", type=click.Path(), help="Output file path")
@click.option("--model", "model", type=str, help="Model ID to use")
@click.option("--language", "language", type=str, help="Document language (e.g., 'ar', 'en')")
@click.option("--document-type", "document_type", type=str, help="Document type (e.g., 'handwritten', 'table')")
@click.option("--quality", "quality", type=click.Choice(['highest', 'high', 'balanced', 'fast']), default='high', help="Quality preference")
@click.option("--format", "format", type=click.Choice(['markdown', 'html', 'json', 'text']), default='markdown', help="Output format")
@click.option("--device", "device", type=click.Choice(['cuda', 'cpu']), default='cuda', help="Device to use")
@click.option("--verbose", "verbose", is_flag=True, help="Verbose output")
def ocr(
    input_path: str,
    output: Optional[str],
    model: Optional[str],
    language: Optional[str],
    document_type: Optional[str],
    quality: str,
    format: str,
    device: str,
    verbose: bool,
):
    """
    Perform OCR on a document with intelligent model selection.
    """
    p = Path(input_path)

    if verbose:
        click.echo(f"Loading document: {p}")

    if p.suffix.lower() == '.pdf':
        images = load_pdf_images(p)
    else:
        images = [load_image(p)]

    if verbose:
        click.echo(f"Loaded {len(images)} page(s)")

    if model:
        if verbose:
            click.echo(f"Loading model: {model}")
        ocr_model = create_model(model, device=device)
    else:
        if verbose:
            click.echo(f"Auto-selecting model (language={language}, type={document_type}, quality={quality})")
        ocr_model = create_recommended_model(
            language=language,
            document_type=document_type,
            quality=quality,
            device=device,
        )
        if verbose:
            click.echo(f"Selected model: {ocr_model.__class__.__name__}")

    results = []
    for i, image in enumerate(images):
        if verbose:
            click.echo(f"Processing page {i+1}/{len(images)}...")
        result = ocr_model.process_image(image)
        results.append(result)

    full_text = "\n\n---\n\n".join([r.text for r in results])

    if output:
        output_path = Path(output)
        if verbose:
            click.echo(f"Saving to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            if format == 'json':
                json.dump({
                    "pages": [
                        {
                            "text": r.text,
                            "confidence": r.confidence,
                            "model": r.model_name,
                            "metadata": r.metadata,
                        }
                        for r in results
                    ]
                }, f, ensure_ascii=False, indent=2)
            else:
                f.write(full_text)
        click.echo(f"✓ Saved to {output_path}")
    else:
        click.echo(full_text)


@cli.command(name="list-available-models")
@click.option("--tier", "tier", type=click.Choice(['tier_1_primary', 'tier_2_specialized', 'tier_3_lightweight', 'tier_4_baseline', 'commercial']), help="Filter by tier")
@click.option("--language", "language", type=str, help="Filter by language support")
@click.option("--commercial-only", "commercial_only", is_flag=True, help="Show only commercial-use models")
@click.option("--json-output", "json_output", is_flag=True, help="Output as JSON")
def list_available_models(tier, language, commercial_only, json_output):
    """List all available OCR models"""
    tier_enum = ModelTier(tier) if tier else None
    models = list_models(tier=tier_enum, language=language, commercial_only=commercial_only)
    if json_output:
        click.echo(json.dumps([m.to_dict() for m in models], indent=2))
    else:
        click.echo("\n" + "="*80)
        click.echo("Manazir OCR - Available Models")
        click.echo("="*80 + "\n")
        for model in models:
            click.echo(f"Model ID: {model.model_id}")
            click.echo(f"  Name: {model.display_name}")
            click.echo(f"  Tier: {model.tier.value}")
            click.echo(f"  Languages: {', '.join(model.languages[:5])}{'...' if len(model.languages) > 5 else ''}")
            click.echo(f"  Strengths: {', '.join(model.strengths)}")
            click.echo(f"  License: {model.license}")
            click.echo(f"  Commercial Use: {'✓' if model.commercial_use else '✗'}")
            click.echo(f"  Description: {model.description}")
            click.echo()


@cli.command(name="recommend")
@click.option("--language", "language", type=str, help="Target language")
@click.option("--document-type", "document_type", type=str, help="Document type")
@click.option("--quality", "quality", type=click.Choice(['highest', 'high', 'balanced', 'fast']), default='high')
def recommend(language, document_type, quality):
    """Get model recommendation based on requirements"""
    from docustruct.model.registry import get_recommended_model, get_model_config
    model_id = get_recommended_model(
        language=language,
        document_type=document_type,
        quality=quality,
    )
    config = get_model_config(model_id)
    click.echo("\n" + "="*80)
    click.echo("Recommended Model")
    click.echo("="*80 + "\n")
    click.echo(f"Model ID: {config.model_id}")
    click.echo(f"Name: {config.display_name}")
    click.echo(f"Description: {config.description}")
    click.echo(f"\nReasoning:")
    if language:
        click.echo(f"  - Language: {language}")
    if document_type:
        click.echo(f"  - Document Type: {document_type}")
    click.echo(f"  - Quality: {quality}")
    click.echo()


@cli.command(name="info")
def info():
    """Show Manazir OCR information"""
    click.echo("\n" + "="*80)
    click.echo("Manazir OCR - Multi-Model OCR Framework")
    click.echo("="*80 + "\n")
    click.echo("Version: 0.2.0")
    click.echo(f"Total Models: {len(MODEL_REGISTRY)}")
    click.echo(f"Tier 1 (Primary): {len([m for m in MODEL_REGISTRY.values() if m.tier == ModelTier.TIER_1_PRIMARY])}")
    click.echo(f"Tier 2 (Specialized): {len([m for m in MODEL_REGISTRY.values() if m.tier == ModelTier.TIER_2_SPECIALIZED])}")
    click.echo(f"Tier 3 (Lightweight): {len([m for m in MODEL_REGISTRY.values() if m.tier == ModelTier.TIER_3_LIGHTWEIGHT])}")
    click.echo(f"Tier 4 (Baseline): {len([m for m in MODEL_REGISTRY.values() if m.tier == ModelTier.TIER_4_BASELINE])}")
    click.echo(f"Commercial APIs: {len([m for m in MODEL_REGISTRY.values() if m.tier == ModelTier.COMMERCIAL])}")
    click.echo("\nFor more information, visit: https://github.com/hesham-haroon/docustruct")
    click.echo()


def main():
    cli()


if __name__ == "__main__":
    main()
