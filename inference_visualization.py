
# Inference Function

def segment_image(model, image_path_or_array, device='cuda'):
    """Perform segmentation on an image"""
    # Preprocess
    image_tensor, (orig_h, orig_w), (new_h, new_w) = preprocess_image(image_path_or_array)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

    # Crop to actual size (remove padding)
    pred = pred[:new_h, :new_w]

    # Resize to original size
    pred = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h),
                     interpolation=cv2.INTER_NEAREST)

    return pred

# Visualization Functions

def visualize_segmentation_with_labels(original_image, segmentation_mask, alpha=0.5):
    """Visualize segmentation result with class labels"""
    # Create colored mask
    colored_mask = COLOR_MAP[segmentation_mask]

    # Load original image if path is provided
    if isinstance(original_image, str):
        original_image = cv2.imread(original_image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Blend original image with mask
    blended = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)

    # Get unique classes present in the image
    unique_classes = np.unique(segmentation_mask)
    present_classes = [cls for cls in unique_classes if cls > 0]  # Exclude background

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(blended)
    title = 'Overlay (Split 3 - Base Classes Only)'
    if present_classes:
        title += f'\nDetected: {", ".join([SPLIT_3_BASE_CLASSES[c] for c in present_classes[:5]])}'
        if len(present_classes) > 5:
            title += '...'
    axes[2].set_title(title, fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add legend with detected classes
    if present_classes:
        legend_elements = []
        for cls in present_classes[:10]:  # Show up to 10 classes
            color = COLOR_MAP[cls] / 255.0
            from matplotlib.patches import Rectangle
            legend_elements.append(Rectangle((0, 0), 1, 1, fc=color,
                                            label=f'{SPLIT_3_BASE_CLASSES[cls]}'))
        if legend_elements:
            axes[2].legend(handles=legend_elements, loc='upper right',
                          fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.show()

    return blended

def print_class_statistics_detailed(segmentation_mask):
    """Print detailed statistics about segmented classes"""
    unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
    total_pixels = segmentation_mask.size

    print("\n" + "="*70)
    print("SEGMENTATION STATISTICS - SPLIT 3")
    print("="*70)
    print(f"{'Class ID':<10} {'Class Name':<20} {'Pixels':<12} {'Percentage':<10}")
    print("-"*70)

    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_name = SPLIT_3_BASE_CLASSES.get(cls, f'Unknown-{cls}')
        print(f"{cls:<10} {class_name:<20} {count:<12,} {percentage:>6.2f}%")

    print("="*70)

    # Summary
    detected_objects = [SPLIT_3_BASE_CLASSES[cls] for cls in unique_classes if cls > 0]
    if detected_objects:
        print(f"\n Detected objects: {', '.join(detected_objects)}")
    else:
        print("\n No objects detected (only background)")
    print()

# Upload and Segment Images

print("\n" + "="*70)
print("Upload an image to segment (Split 3 - Base Classes):")
print("="*70)
uploaded_images = files.upload()

for filename, data in uploaded_images.items():
    print(f"\n{'='*70}")
    print(f"Processing: {filename}")
    print('='*70)

    # Read image from uploaded data
    image_array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

    # Perform segmentation
    print("Running segmentation...")
    segmentation_result = segment_image(model, image, device)

    # Visualize results with enhanced labels
    print("Visualizing results...")
    blended = visualize_segmentation_with_labels(image, segmentation_result, alpha=0.5)

    # Print detailed statistics
    print_class_statistics_detailed(segmentation_result)

    # Save result
    output_filename = f"segmented_split3_{filename}"
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_filename, blended_bgr)
    print(f"💾 Saved result to: {output_filename}")
    print('='*70)


#  Batch Processing

def batch_segment_images(model, image_list, device='cuda'):
    """Process multiple images at once"""
    results = []
    for i, img in enumerate(image_list):
        print(f"Processing image {i+1}/{len(image_list)}...")
        pred = segment_image(model, img, device)
        results.append(pred)
    return results


# To download all results:
# import zipfile
# import os
#
# zip_filename = 'segmentation_results.zip'
# with zipfile.ZipFile(zip_filename, 'w') as zipf:
#     for file in os.listdir('.'):
#         if file.startswith('segmented_split3_'):
#             zipf.write(file)
#
# colab_files.download(zip_filename)
