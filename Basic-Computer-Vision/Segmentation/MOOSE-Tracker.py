"""

MOOSE - Minimum Output Sum Of Squared Error (Filter)

This is proposed by for fast object tracking using correlation filter methods. Correlation
filter-based tracking comprises of:

1. Assuming a template of a target object T and an input image I, we first take the Fast
Fourier Transform(FFT) of both the template (T) and the image (I).

2. A convolution operation is performed between template T and image I.

3. The result from step 2 is inverted to the spatial domain using Inverse Fast
Fourier Transform (IFFT). The position of the template object in the image I is the
max value of the IFFT response we get.


"""

def correlate(self, img):
 """
 Correlation of input image with the kernel
 """
 # get response in fourier domain
 C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT),
 self.H, 0, conjB=True)
 # compute inverse to get image domain output
 resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

 # max location of the response
 h, w = resp.shape
 _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
 side_resp = resp.()
 cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
 smean, sstd = side_resp.mean(), side_resp.std()
 psr = (mval-smean) / (sstd+eps)
 # displacement of max location from center is displacement for
 tracker
 return resp, (mx-w//2, my-h//2), psr

# The update function gets a frame from video or image sequence iteratively and updates the
# state of the tracker.

def update(self, frame, rate = 0.125):
 # compute current state and window size
 (x, y), (w, h) = self.pos, self.size

 # compute and update rectangular area from new frame
 self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))

 # pre-process it by normalization
 img = self.preprocess(img)

 # apply correlation and compute displacement
 self.last_resp, (dx, dy), self.psr = self.correlate(img)

 self.good = self.psr > 8.0
 if not self.good:
 return
 # update pos
 self.pos = x+dx, y+dy
 self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
 img = self.preprocess(img)

 A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
 H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
 H2 = cv2.mulSpectrums( A, A, 0, conjB=True)

 self.H1 = self.H1 * (1.0-rate) + H1 * rate
 self.H2 = self.H2 * (1.0-rate) + H2 * rate
 self.update_kernel()


"""

A major advantage of using the MOSSE filter is that it is quite fast for real-time tracking
systems.
The overall algorithm is simple to implement and can be used in the hardware
without special image processing libraries, such as embedded platforms.
There have been several modifications to this filter and, as such, readers are requested to explore more about
these filters.

"""