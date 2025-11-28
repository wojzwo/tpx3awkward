def centroid_to_canvas_subpixel_full(
		df: pd.DataFrame,
		scale: int = 1,
		x_col: str = "xc",
		y_col: str = "yc",
		weight: str = "count",
		canvas: np.ndarray | None = None,
		offset_x: float = 0.0,
		offset_y: float = 0.0,
		renormalize: bool = True,
		dtype=np.float32,
):
	"""
	Distribute each centroid over up to 4 pixels using bilinear (area) weights so
	that the total intensity contributed by a centroid equals its weight.

	For centroid with scaled coordinate (x_s, y_s):
	  floor/ceil in x,y -> up to 4 pixels
	  fx = frac(x_s), fy = frac(y_s)
	  Weights: (1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy

	Parameters
	----------
	df : DataFrame with centroid positions.
	scale : linear scale (1 => 256x256).
	weight : 'count' or column name.
	renormalize : if some pixels fall outside canvas, remaining weights are rescaled to preserve total.
	"""
	if scale < 1:
		raise ValueError("scale must be >=1")
	size = 256 * scale

	size_scaled=256*scale
	if canvas is None:
		canvas = np.zeros((size, size), dtype=dtype)

	x = (df[x_col].to_numpy(dtype=np.float64) + offset_x) * scale
	y = (df[y_col].to_numpy(dtype=np.float64) + offset_y) * scale

	n = x.size
	if weight == "count":
		base_w = np.ones(n, dtype=np.float64)
	else:
		if weight not in df.columns:
			raise ValueError(f"weight column '{weight}' not found")
		base_w = df[weight].to_numpy(dtype=np.float64)

	floor_x = np.floor(x)
	floor_y = np.floor(y)
	mx= x-floor_x
	my= y-floor_y
	# If fractional part is 0 exactly, we keep single pixel
	x1=floor_x.astype(np.int32)
	x2=x1+1
	y1=floor_y.astype(np.int32)
	y2=y1+1
	x2[x2>=size_scaled]=size_scaled-1
	y2[y2>=size_scaled]=size_scaled-1
	w11=(1 - mx) * (1 - my)*base_w
	w12=(1 - mx) * my*base_w
	w21=mx * (1 - my)*base_w
	w22=mx * my*base_w

	cx = np.concatenate([x1, x1, x2, x2])
	cy = np.concatenate([y1, y2, y1, y2])
	w  = np.concatenate([w11, w12, w21, w22])

	idx = np.repeat(np.arange(n), 4)

	canvas=np.zeros((size_scaled, size_scaled), dtype=np.float64)


	np.add.at(canvas, (cy, cx), w)
	return canvas

