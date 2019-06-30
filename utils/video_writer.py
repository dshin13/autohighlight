# Dependencies
import cv2
import numpy as np

# video exporter to collect frames, process them with overlay and output to file
# Working fourcc codec/extension: 'MP4V', '.mp4'

def write_to_video(src, outdir, fps=25.0):
    """Video exporter to write source frame stack into video
    Defaults to writing a video file with '.mp4' extension
    
    Parameters
    ----------
    src : numpy.array
        Source of shape (frames x H x W x C)
    outdir : str
        Output file path
    fps : float
        Video frame rate (frames per second)
    """
    
    # if source is in float32, convert to uint8
    if src.dtype == 'float32':
        src = (src * 255).astype('uint8')
    
    size = (src.shape[2], src.shape[1])
    
    frames = src.shape[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    
    out = cv2.VideoWriter(outdir, fourcc, fps, size)
    
    for f in range(frames):
        out.write(src[f])

# overlay function
def overlay_images(source, overlay, alpha):
    """Overlays source image with the provided overlay image
    
    Parameters
    ----------
    source : numpy.array
        Source image
    overlay : numpy.array
        Overlay image
    alpha : float
        Alpha value (fraction of overlay to combine with source)
        Source image is scaled by factor of (1 - alpha) before it is added
    
    Returns
    -------
        numpy.array : returns source combined with overlay image
    """
    
    output = source.copy()
        
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

# helper function to match frames and shapes
def match_frames_and_dims(input, template, overlay=False, alpha=0.5):
    """Matches frames and dimensions of input to that of template
    
    Parameters
    ----------
    input : numpy.array
        Input video
    template : numpy.array
        Template video - provides target frame count and dimensions
    overlay : bool
        Flag to determine whether to perform overlay of source directly
        over template
    alpha : float
        Alpha value (fraction of input to combine with template)
    
    Returns
    -------
        numpy.array : dimension-matched input (or input combined with
        template if overlay flag is set to True)
    """
    
    # (Optional) fractional power of input to increase contrast
    input = np.power(input, 0.8)
    
    num_frames, h, w, num_channels = template.shape
    input_frames = input.shape[0]
    print(template.shape)
    
    # if template is in 0-255 scale, convert into float
    if template.dtype == 'uint8':
        template = (template / 255).astype('float32')
    
    # otherwise, scale to range (0,1)
    else:
        template = (template - template.min()) / (template.max() - template.min())
        template = template.astype('float32')
    
    # Step 1: match number of frames by duplicating channels until filled
    num_duplicates = round(num_frames / input_frames)
    output = np.repeat(input, num_duplicates, axis=0) # fill duplicate frames
    print(output.shape)

    # if using 1-dimensional data, copy contents into missing channels
    if len(input.shape) == 3:
        output = np.stack([output, output, output], axis=-1)
    
    # Step 2: resize each frame to match template dimensions
    output = [output[n] for n in range(num_frames)] # truncate excess frames to match frame count
    for n in range(num_frames):
        output[n] = cv2.resize(output[n], (w, h))
        
        if overlay:
            output[n] = overlay_images(template[n], output[n], alpha)
    
    output = np.array(output)
    
    return output

if __name__ == '__main__':
    # typical usage pattern:
    # First generate a frame stack using match_frames_and_dims function
    # Then write it to file
    overlay_video = match_frames_and_dims(o_stack, img_stack, overlay=True, alpha=0.5)
    write_to_video(overlay_video, 'overlay_video.mp4')
