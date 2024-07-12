Place in this folder any image that you want to use as a custom size.
By default, png is supported, no guarantee is made for other formats.

The image should be of adequate size to handle any up-/downscaling that you may request later 
on through the agent's vision size.
Additionally, given w and h, the image's dimensions, these should be either:
 - [equal] square image, image will be rotated when necessary)
 - [4w = h] rectangular, every square slice will used for a specific orientation in the
   following order: West, North, East and South.
