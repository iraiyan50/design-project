
# ============================================================================
# CELL 5: Load Model
# ============================================================================

def load_model(checkpoint_path, device='cuda'):
    """Load the trained model from checkpoint"""
    print("Creating model...")
    model = PSPNet(
        num_classes=NUM_CLASSES,
        layers=LAYERS,
        bins=BINS,
        bottleneck_dim=BOTTLENECK_DIM,
        dropout=DROPOUT,
        m_scale=M_SCALE
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model

# Uploading model file
print("\n" + "="*70)
print("Please upload your Split 3 .pth checkpoint file:")
print("="*70)
uploaded = files.upload()

# Geting the uploaded file name
checkpoint_path = list(uploaded.keys())[0]

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

model = load_model(checkpoint_path, device)

# ============================================================================
# CELL 6: Preprocessing Functions
# ============================================================================

def preprocess_image(image_path_or_array, image_size=IMAGE_SIZE):
    """Preprocess image for model input"""
    # Load image
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_path_or_array

    original_h, original_w = image.shape[:2]

    # Resize keeping aspect ratio
    if original_h >= original_w:
        ratio = image_size / original_h
        new_h = image_size
        new_w = int(original_w * ratio)
    else:
        ratio = image_size / original_w
        new_h = int(original_h * ratio)
        new_w = image_size

    # Make dimensions divisible by 8
    new_h = (new_h // 8) * 8
    new_w = (new_w // 8) * 8

    # Resize image
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    padded_image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    padded_image[:new_h, :new_w, :] = image_resized

    # Normalize
    padded_image = padded_image / 255.0
    for i in range(3):
        padded_image[:, :, i] = (padded_image[:, :, i] - MEAN[i]) / STD[i]

    # Convert to tensor
    image_tensor = torch.from_numpy(padded_image.transpose(2, 0, 1)).float()
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, (original_h, original_w), (new_h, new_w)


