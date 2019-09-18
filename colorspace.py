# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:19:08 2019

Colorspace conversion to/from HSL

Last Modified: 17/8/2019

@author: Rivan
"""

import math

def rgb2hsl(red,green,blue):
  """
  Convert color from RGB to HSL colorspace
  Input: RGB [0..1]
  Output: HSL [0..1]
  Ref: https://en.wikipedia.org/wiki/HSL_and_HSV
  
  """
  # Normalize
  if (red > 1 or green > 1 or blue > 1):
      red = red/255.
      green = green/255.
      blue = blue/255.
  
  high = max(red,green,blue)
  low = min(red,green,blue)
  
  luminance = (high + low)/2.
  
  if (high == low):
    hue = saturation = 0 # achromatic
  else:
    d = high - low
    saturation = d / (2 - high - low) if luminance > 0.5 else d / (high + low)
    hue = {
            red: (green - blue) / d + (6 if green < blue else 0),
            green: (blue - red) / d + 2,
            blue: (red - green) / d + 4,      
    }[high]
    hue /= 6
    
  return hue, saturation, luminance


def hsl2rgb(hue, saturation, luminance):
  """
  Convert color from HSL to RGB colorspace
  Input: HSL [0..1]
  Output: RGB [0..255]
  Ref: https://en.wikipedia.org/wiki/HSL_and_HSV
  
  """  
  if (saturation == 0):
    red = green = blue = luminance # achromatic
  else:
    def hue2rgb(p, q, t):
        t += 1 if t < 0 else 0
        t -= 1 if t > 1 else 0
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q          
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
      
    q = luminance * (1 + saturation) if luminance < 0.5 else luminance + saturation - luminance * saturation
    p = 2 * luminance - q
    
    red = hue2rgb(p,q, hue + 1/3)
    green = hue2rgb(p,q, hue)
    blue = hue2rgb(p,q, hue - 1/3)
    
  return round(red * 255.), round(green * 255.), round(blue * 255.)


def hsv2hsl(hue, saturation, value):
    """
      Convert color from HSV to HSL colorspace
      Input: HSV [0..1]
      Output: HSL [0..1]
      Ref: https://en.wikipedia.org/wiki/HSL_and_HSV
  
    """ 
    luminance = 0.5 * value  * (2 - saturation)
    saturation = value * saturation / (1 - math.fabs(2*luminance-1))
    return hue, saturation, luminance

def hsl2hsv(hue, saturation, luminance):
    """
      Convert color from HSL to HSV colorspace
      Input: HSL [0..1]
      Output: HSV [0..1]
      Ref: https://en.wikipedia.org/wiki/HSL_and_HSV
  
    """ 
    value = (2*luminance + saturation*(1-math.fabs(2*luminance-1)))/2
    saturation = 2*(value-luminance)/value
    return hue, saturation, value





