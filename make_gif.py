import os
import imageio.v2 as imageio

# ============================
# User settings
# ============================
img_dir = "/Users/YPL/Documents/Experiments/myExp/pics/"  # directory containing the images
output_gif = "XXXXX.gif"
fps = 10                  # frames per second
ext = ".png"             # image extension
prefix = ""   # file name prefix
# ============================


def make_gif(img_dir, output_gif, fps=fps, prefix=None, ext=".png"):
    images = []

    files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(ext) and (prefix is None or f.startswith(prefix))
    ])

    if len(files) == 0:
        raise RuntimeError("No images found to make GIF.")

    print(f"Found {len(files)} images")

    for f in files:
        img_path = os.path.join(img_dir, f)
        images.append(imageio.imread(img_path))

    imageio.mimsave(
        os.path.join("/Users/YPL/Documents/Experiments/myExp", output_gif),
        images,
        fps=fps
    )
    

    print(f"GIF saved to: {os.path.join('/Users/YPL/Documents/Experiments/myExp', output_gif)}")


if __name__ == "__main__":
    make_gif(
        img_dir=img_dir,
        output_gif=output_gif,
        fps=fps,
        prefix=prefix,
        ext=ext
    )
