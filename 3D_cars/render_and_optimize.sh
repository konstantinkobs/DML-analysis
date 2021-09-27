# In order to run this, you need Blender and Docker installed on your system.
# Alternatively to Docker, you can discard the optimization or use ImageOptim directly.

# Create directories
mkdir -p '/data/renders'

# Render images
cd data
blender -y -b 'cars.blend' -o 'renders/' -f 1..100000 -- --cycles-device CUDA

# Optimize images
# (This step is optional, but Blender blows up all rendered images and this step minimizes these images without quality loss)
docker run -v ./renders/:/images/src jahvi/imageoptim
