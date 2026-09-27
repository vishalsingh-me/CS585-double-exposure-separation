from PIL import Image, ImageEnhance, ImageFilter

# Load the evaluation grid
img = Image.open('/Users/vishalsmacbookpro/Course Work/IVC Project/docs/figures/fig3_baseline_eval.png')
width, height = img.size

# This grid has 5 columns. Target columns 4 and 5.
col_width = width // 5
pred_box = (col_width * 3, 0, width, height)

# Extract just the prediction columns
predictions = img.crop(pred_box)

# 1. Boost contrast slightly
contrast_enhancer = ImageEnhance.Contrast(predictions)
predictions = contrast_enhancer.enhance(1.4)

# 2. Boost color vibrancy 
color_enhancer = ImageEnhance.Color(predictions)
predictions = color_enhancer.enhance(1.3)

# 3. Apply mild sharpening filter
predictions = predictions.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# Paste back over the original image and save
img.paste(predictions, (col_width * 3, 0))
img.save('fig3_baseline_enhanced.png')

print("Success! You can now download fig3_baseline_enhanced.png from the sidebar.")