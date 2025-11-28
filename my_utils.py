from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile, ParquetWriter

import numpy as np
import gc
from numpy.typing import NDArray
import numba
import math

try:
	from tqdm.notebook import tqdm  # prettier in Jupyter
except Exception:
	from tqdm.auto import tqdm  # fallback

from IPython.core.interactiveshell import InteractiveShell

from tpx3awkward._utils import (
	drop_zero_tot,
	cluster_df_optimized,
	group_indices,
	centroid_clusters,
	ingest_cent_data,
	add_centroid_cols,
	trim_corr_file,
	trim_corr,
	timewalk_corr,
	DEFAULT_CLUSTER_TW,
	DEFAULT_CLUSTER_RADIUS,
	raw_as_numpy,
	ingest_raw_data,
	is_packet_header,
	matches_nibble,
	get_block,
	decode_message,
)

DEFAULT_CLUSTER_RADIUS = 3
DEFAULT_CLUSTER_TW = int(0.3e-6 / 1.5625e-9)  # reuse your constant
DEFAULT_CLUSTER_TW


@numba.njit(cache=True)
def cluster_events(events: np.ndarray, radius: int, tw: int):
	# events: (N,4) -> x,y,ToT,t
	n = events.shape[0]
	labels = np.full(n, -1, np.int64)
	cluster_id = 0
	r2 = radius * radius
	for i in range(n):
		if labels[i] != -1:
			continue
		labels[i] = cluster_id
		t_i = events[i, 3]
		x_i = events[i, 0]
		y_i = events[i, 1]
		for j in range(i + 1, n):
			# early exit on time window
			if events[j, 3] - t_i > tw:
				break
			dx = x_i - events[j, 0]
			dy = y_i - events[j, 1]
			if dx * dx + dy * dy <= r2:
				labels[j] = cluster_id
		cluster_id += 1
	return labels


def cluster_df_corrected(df: pd.DataFrame, tw=DEFAULT_CLUSTER_TW, radius=DEFAULT_CLUSTER_RADIUS):
	df_sorted = df.sort_values("t", kind="mergesort").reset_index(drop=True)
	events = df_sorted[["x", "y", "ToT", "t"]].to_numpy()
	labels = cluster_events(events, radius, tw)
	return labels, events


@numba.njit(cache=True)
def group_indices(labels: np.ndarray):
	n_clusters = labels.max() + 1
	counts = np.zeros(n_clusters, np.int32)
	for lab in labels:
		counts[lab] += 1
	max_size = counts.max()
	out = -1 * np.ones((n_clusters, max_size), dtype=np.int32)
	fill = np.zeros(n_clusters, np.int32)
	for i in range(labels.size):
		c = labels[i]
		pos = fill[c]
		out[c, pos] = i
		fill[c] += 1
	return out


@numba.njit(cache=True)
def centroid_clusters(cluster_arr: np.ndarray, events: np.ndarray):
	n_clusters = cluster_arr.shape[0]
	max_cluster = cluster_arr.shape[1]
	t = np.zeros(n_clusters, dtype=np.uint64)
	xc = np.zeros(n_clusters, dtype=np.float32)
	yc = np.zeros(n_clusters, dtype=np.float32)
	ToT_max = np.zeros(n_clusters, dtype=np.uint32)
	ToT_sum = np.zeros(n_clusters, dtype=np.uint32)
	n = np.zeros(n_clusters, dtype=np.uint8)
	for cid in range(n_clusters):
		local_max = np.uint32(0)
		for k in range(max_cluster):
			ei = cluster_arr[cid, k]
			if ei < 0:
				break
			x = events[ei, 0]
			y = events[ei, 1]
			tot = np.uint32(events[ei, 2])
			ts = events[ei, 3]
			if tot > local_max:
				local_max = tot
				ToT_max[cid] = tot
				t[cid] = ts
			xc[cid] += x * tot
			yc[cid] += y * tot
			ToT_sum[cid] += tot
			n[cid] += 1
		if ToT_sum[cid] > 0:
			xc[cid] /= ToT_sum[cid]
			yc[cid] /= ToT_sum[cid]
	return t, xc, yc, ToT_max, ToT_sum, n


def add_centroid_cols(df: pd.DataFrame, gap=False):
	if gap:
		df.loc[df["xc"] >= 255.5, "xc"] += 2
		df.loc[df["yc"] >= 255.5, "yc"] += 2
	df["x"] = np.round(df["xc"]).astype(np.uint16)
	df["y"] = np.round(df["yc"]).astype(np.uint16)
	df["t_ns"] = df["t"].astype(np.float64) * 1.5625
	return df


def read_uint64_chunks(fpath, chunk_words=5_000_000):
	fpath = Path(fpath)
	with fpath.open("rb") as fh:
		while True:
			arr = np.fromfile(fh, dtype="<u8", count=chunk_words)
			if arr.size == 0:
				break
			yield arr


class StreamingDecoder:
	def __init__(self):
		self.heartbeat_lsb = None
		self.heartbeat_msb = None
		self.heartbeat_time = np.uint64(0)
		self.hb_init_flag = False
		self.chip_indx = np.uint8(0)

	def process_chunk(self, words: np.ndarray):
		photons = []
		for msg in words:
			if is_packet_header(msg):
				self.chip_indx = np.uint8(get_block(msg, 8, 32))
			elif matches_nibble(msg, 0xB):
				x, y, ToT, t = decode_message(msg, self.chip_indx, heartbeat_time=self.heartbeat_time)
				photons.append((t, x, y, ToT, self.chip_indx))
			elif matches_nibble(msg, 0x4):
				sub = (msg >> np.uint64(56)) & np.uint64(0x0F)
				if sub == 0x4:
					self.heartbeat_lsb = (msg >> np.uint64(16)) & np.uint64(0xFFFFFFFF)
				elif sub == 0x5 and self.heartbeat_lsb is not None:
					self.heartbeat_msb = ((msg >> np.uint64(16)) & np.uint64(0xFFFF)) << np.uint64(32)
					self.heartbeat_time = self.heartbeat_msb | self.heartbeat_lsb
		return photons


def flush_cluster_centroid(df_raw_batch, tw=DEFAULT_CLUSTER_TW, radius=DEFAULT_CLUSTER_RADIUS, gap_corr=False):
	if df_raw_batch.empty:
		# Return empty raw and centroid dfs
		return df_raw_batch, pd.DataFrame(columns=["t", "xc", "yc", "ToT_max", "ToT_sum", "n"])
	df = drop_zero_tot(df_raw_batch)
	labels, events = cluster_df_optimized(df, tw=tw, radius=radius)
	df["cluster_id"] = labels
	cluster_array = group_indices(labels)
	cent_tuple = centroid_clusters(cluster_array, events)
	cent_df = pd.DataFrame(ingest_cent_data(cent_tuple)).sort_values("t").reset_index(drop=True)
	cent_df = add_centroid_cols(cent_df, gap=gap_corr)  # was gap=True
	return df, cent_df


def stream_process_file(
		fpath,
		out_raw="raw_big.parquet",
		out_cent="cent_big.parquet",
		chunk_words=5_000_000,
		flush_after=2_500_000,
		overlap_ticks=300_000,
		tw=DEFAULT_CLUSTER_TW,
		radius=DEFAULT_CLUSTER_RADIUS,
		gap_corr=False,  # new parameter
):
	"""
	Streaming decode + clustering with incremental (append) parquet writes.
	Fixes previous behavior that overwrote the parquet at every flush.
	"""
	raw_out = Path(out_raw)
	cent_out = Path(out_cent)

	# Start clean
	if raw_out.exists():
		raw_out.unlink()
	if cent_out.exists():
		cent_out.unlink()

	decoder = StreamingDecoder()
	buffer = []
	max_t = 0

	# For incremental parquet writing
	raw_writer = None
	cent_writer = None

	fsize = Path(fpath).stat().st_size
	est_chunks = math.ceil(fsize / (chunk_words * 8)) or None

	photons_total = 0
	pbar = tqdm(
		read_uint64_chunks(fpath, chunk_words=chunk_words),
		total=est_chunks,
		desc="Streaming decode",
		unit="chunk"
	)

	def write_batch(raw_df: pd.DataFrame, cent_df: pd.DataFrame):
		nonlocal raw_writer, cent_writer
		# Convert once per batch
		raw_table = pa.Table.from_pandas(raw_df, preserve_index=False)
		cent_table = pa.Table.from_pandas(cent_df, preserve_index=False)
		if raw_writer is None:
			raw_writer = ParquetWriter(str(raw_out), raw_table.schema, compression="zstd")
		if cent_writer is None:
			cent_writer = ParquetWriter(str(cent_out), cent_table.schema, compression="zstd")
		raw_writer.write_table(raw_table)
		cent_writer.write_table(cent_table)

	for chunk in pbar:
		photons = decoder.process_chunk(chunk)
		if photons:
			buffer.extend(photons)
			photons_total += len(photons)
			mt = max(p[0] for p in photons)
			if mt > max_t:
				max_t = mt

		pbar.set_postfix({"photons": photons_total, "buffer": len(buffer)})

		if len(buffer) >= flush_after:
			buffer.sort(key=lambda r: r[0])
			cutoff = max_t - overlap_ticks
			flush_part = [p for p in buffer if p[0] <= cutoff]
			buffer = [p for p in buffer if p[0] > cutoff]

			if flush_part:
				df_flush = pd.DataFrame(flush_part, columns=["t", "x", "y", "ToT", "chip"])
				raw_df, cent_df = flush_cluster_centroid(df_flush, tw=tw, radius=radius, gap_corr=gap_corr)
				write_batch(raw_df, cent_df)
				del raw_df, cent_df, df_flush, flush_part
				gc.collect()

	# Final flush
	if buffer:
		buffer.sort(key=lambda r: r[0])
		df_final = pd.DataFrame(buffer, columns=["t", "x", "y", "ToT", "chip"])
		raw_df, cent_df = flush_cluster_centroid(df_final, tw=tw, radius=radius, gap_corr=gap_corr)
	else:
		raw_df = pd.DataFrame(columns=["t", "x", "y", "ToT", "chip", "cluster_id"])
		cent_df = pd.DataFrame(columns=["t", "xc", "yc", "ToT_max", "ToT_sum", "n", "x", "y", "t_ns"])

	write_batch(raw_df, cent_df)

	# Close writers
	if raw_writer is not None:
		raw_writer.close()
	if cent_writer is not None:
		cent_writer.close()

	return raw_out, cent_out


def make_scaled_canvas(scale: int, dtype=np.uint32):
	"""
	Allocate a blank canvas for a scaled 256x256 sensor.
	Final shape: (256*scale, 256*scale)
	"""
	if scale < 1:
		raise ValueError("scale must be >= 1")
	size = 256 * scale
	return np.zeros((size, size), dtype=dtype)


def load_parquet_range(path: str | Path,
                       i_min: int | None = None,
                       i_max: int | None = None,
                       columns: list[str] | None = None) -> pd.DataFrame:
	"""
	Load a row slice [i_min, i_max) from a parquet file without reading all data.
	If i_min or i_max is None, the whole file (optionally restricted to `columns`) is loaded.

	Parameters
	----------
	path : str | Path
		 Parquet file path.
	i_min : int | None
		 Start row (inclusive). None -> full file.
	i_max : int | None
		 End row (exclusive). None -> full file.
	columns : list[str] | None
		 Column subset to read. None -> all columns.

	Returns
	-------
	pd.DataFrame
	"""
	pf = ParquetFile(str(path))

	# Full read path
	if i_min is None or i_max is None:
		# Efficient full read (optionally select columns)
		tables = []
		for rg in range(pf.num_row_groups):
			tables.append(pf.read_row_group(rg, columns=columns))
		if not tables:
			return pd.DataFrame()
		return pa.concat_tables(tables).to_pandas()

	if i_min >= i_max or i_max <= 0:
		return pd.DataFrame()

	out = []
	offset = 0
	for rg in range(pf.num_row_groups):
		n_rows = pf.metadata.row_group(rg).num_rows
		rg_start = offset
		rg_end = offset + n_rows

		if rg_end <= i_min:  # row group before range
			offset = rg_end
			continue
		if rg_start >= i_max:  # past requested range
			break

		inner_start = max(0, i_min - rg_start)
		inner_end = min(n_rows, i_max - rg_start)

		table = pf.read_row_group(rg, columns=columns)
		if inner_start != 0 or inner_end != n_rows:
			table = table.slice(inner_start, inner_end - inner_start)
		out.append(table)
		offset = rg_end

	if not out:
		return pd.DataFrame()

	return pa.concat_tables(out).to_pandas()


def centroid_to_canvas(
		df: pd.DataFrame,
		scale: int = 1,
		x_col: str = "xc",
		y_col: str = "yc",
		weight: str = "count",
		canvas: np.ndarray | None = None,
		offset_x: float = 0.0,
		offset_y: float = 0.0,
):
	"""
	Project centroid rows onto a scaled pixel canvas using mathematical rounding.

	Parameters
	----------
	df : pd.DataFrame       DataFrame with centroid columns.
	scale : int             Linear scale factor (1 -> 256x256, 2 -> 512x512, ...).
	x_col, y_col : str      Columns holding centroid coordinates (float).
	weight : str            'count' -> increment per centroid, or name of a numeric column (e.g. 'ToT_sum').
	canvas : np.ndarray     Optional preallocated target. If None, a new one is made.
	offset_x, offset_y : float  Shifts applied before rounding/scaling.

	Returns
	-------
	np.ndarray
		 Accumulated canvas.
	"""
	if canvas is None:
		canvas = make_scaled_canvas(scale)
	size = 256 * scale

	# Extract coordinates
	x = df[x_col].to_numpy(dtype=np.float64)
	y = df[y_col].to_numpy(dtype=np.float64)

	# Apply scaling and rounding (mathematical: round half away from zero via np.rint)
	x_pix = np.rint((x + offset_x) * scale).astype(np.int64)
	y_pix = np.rint((y + offset_y) * scale).astype(np.int64)

	# Bounds mask
	m = (x_pix >= 0) & (x_pix < size) & (y_pix >= 0) & (y_pix < size)
	if not np.any(m):
		return canvas

	if weight == "count":
		vals = np.ones(m.sum(), dtype=canvas.dtype)
	else:
		if weight not in df.columns:
			raise ValueError(f"weight column '{weight}' not in DataFrame")
		vals = df.loc[m, weight].to_numpy()
		# Cast / sanitize
		if not np.issubdtype(vals.dtype, np.number):
			raise TypeError(f"weight column '{weight}' must be numeric")
		if canvas.dtype != vals.dtype:
			vals = vals.astype(canvas.dtype, copy=False)

	# Accumulate (y first because row-major)
	np.add.at(canvas, (y_pix[m], x_pix[m]), vals)
	return canvas


def build_scaled_canvases(
		df: pd.DataFrame,
		scales: list[int],
		weight: str = "count",
		x_col: str = "xc",
		y_col: str = "yc",
		offset_x: float = 0.0,
		offset_y: float = 0.0,
):
	"""
	Convenience: build multiple scaled canvases.
	Returns dict scale -> canvas.
	"""
	out = {}
	for s in scales:
		c = make_scaled_canvas(s)
		centroid_to_canvas(
			df=df,
			scale=s,
			x_col=x_col,
			y_col=y_col,
			weight=weight,
			canvas=c,
			offset_x=offset_x,
			offset_y=offset_y,
		)
		out[s] = c
	return out




def centroid_to_canvas_subpixel(
		df: pd.DataFrame,
		scale: int = 1,
		x_col: str = "xc",
		y_col: str = "yc",
		weight: str = "count",
		canvas: np.ndarray | None = None,
		offset_x: float = 0.0,
		offset_y: float = 0.0,
		chunk_size: int | None =None,
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

	size_scaled = 256 * scale

	canvas = np.zeros((size_scaled, size_scaled), dtype=np.float64)
	n_total = len(df)



	if chunk_size:
		chunk_iter =tqdm(
			range(0, n_total, chunk_size),
			total=(n_total + chunk_size - 1) // chunk_size,
			desc="Aggregating",
			unit="rows",
	)
	else:
		chunk_iter = range(1)

	for start in chunk_iter:
		if chunk_size:
			end = min(start + chunk_size, n_total)
			df_chunk = df.iloc[start:end]
		else:
			df_chunk = df

		x = (df_chunk[x_col].to_numpy(dtype=np.float64) + offset_x) * scale
		y = (df_chunk[y_col].to_numpy(dtype=np.float64) + offset_y) * scale

		if weight == "count":
			base_w = np.ones(x.size, dtype=np.float64)
		else:
			if weight not in df_chunk.columns:
				raise ValueError(f"weight column '{weight}' not found")
			base_w = df_chunk[weight].to_numpy(dtype=np.float64)

		floor_x = np.floor(x)
		floor_y = np.floor(y)
		mx = x - floor_x
		my = y - floor_y

		x1 = floor_x.astype(np.int32)
		y1 = floor_y.astype(np.int32)
		x2 = x1 + 1
		y2 = y1 + 1

		# Clamp to canvas bounds
		x1 = np.clip(x1, 0, size_scaled - 1)
		y1 = np.clip(y1, 0, size_scaled - 1)
		x2 = np.clip(x2, 0, size_scaled - 1)
		y2 = np.clip(y2, 0, size_scaled - 1)

		w11 = (1 - mx) * (1 - my) * base_w
		w12 = (1 - mx) * my * base_w
		w21 = mx * (1 - my) * base_w
		w22 = mx * my * base_w

		cx = np.concatenate([x1, x1, x2, x2])
		cy = np.concatenate([y1, y2, y1, y2])
		w = np.concatenate([w11, w12, w21, w22])

		np.add.at(canvas, (cy, cx), w)

	return canvas


def load_and_centroid_to_canvas_subpixel(path: str | Path,
                                         scale: int = 1,
													  x_col: str = "xc",
													  y_col: str = "yc",
													  weight: str = "count",
													  offset_x: float = 0.0,
													  offset_y: float = 0.0,
                                         chunk_size: int = 1_000_000,
													  ) -> np.ndarray:
	if scale < 1:
		raise ValueError("scale must be >=1")
	size = 256 * scale

	size_scaled = 256 * scale

	canvas = np.zeros((size_scaled, size_scaled), dtype=np.float64)


	pf = ParquetFile(str(path))
	n_total = pf.metadata.num_rows

	for start in tqdm(
			range(0, n_total, chunk_size),
			total=(n_total + chunk_size - 1) // chunk_size,
			desc="Aggregating",
			unit="rows",
	):
		end = min(start + chunk_size, n_total)
		rg_start = start
		rg_end = end
		# Load chunk
		df_chunk = load_parquet_range(path, i_min=rg_start, i_max=rg_end)
		df_chunk['x'] = df_chunk['x'] - 256
		df_chunk['xc'] = df_chunk['xc'] - 256

		# Process chunk
		canvasT = centroid_to_canvas_subpixel(
			df=df_chunk,
			scale=scale,
			x_col=x_col,
			y_col=y_col,
			weight=weight,
			canvas=canvas,
			offset_x=offset_x,
			offset_y=offset_y,
			chunk_size=None,
		)
		canvas+=canvasT
	return canvas


