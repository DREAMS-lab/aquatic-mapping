#!/usr/bin/env python3

from pathlib import Path

from collections import deque

import numpy as np

import threading



try:

    from PyQt5 import QtWidgets, QtCore

    import pyqtgraph as pg



    # Force software rendering to avoid blank/white grabs on some GPUs/VMs

    pg.setConfigOptions(useOpenGL=False, antialias=True)

    _PG_AVAILABLE = True

except ImportError:

    _PG_AVAILABLE = False

    print("WARNING: PyQt5/pyqtgraph not available. Visualization disabled.")



class PGLiveViz:

    def __init__(self, W, H, xs, ys, title_suffix=""):

        self.W, self.H = float(W), float(H)

        self.xs, self.ys = xs, ys

        self.nx, self.ny = len(xs), len(ys)

        self._lock = threading.Lock()

        self._latest = None

        self.planner_id = 1

        self._save_requested = False

        self._save_path = None



        # Fixed color levels (can be overridden via set_levels() from planner)

        self.levels_mu  = (20.0, 30.0)

        self.levels_std = (0.0, 5.0)

        self.levels_mi  = (0.0, 3.0)



        if not _PG_AVAILABLE:

            print("WARNING: Running in headless mode (no visualization)")

            self.app = None

            self.timer = None

            return



        self.app = pg.mkQApp(f"MI Viz {title_suffix}")



        # ---- Window 1: Field Maps (plot + colorbar pairs in columns) ----

        self.win_fields = pg.GraphicsLayoutWidget(title=f"Fields - {title_suffix}")

        self.win_fields.resize(1800, 600)

        self.win_fields.show()



        # --- colormaps (high contrast across panels) ---
        cm_mean   = pg.colormap.get('turbo').getLookupTable(0.0, 1.0, 256)     # Mean
        cm_uncert = pg.colormap.get('inferno').getLookupTable(0.0, 1.0, 256)   # Uncertainty
        cm_mi     = pg.colormap.get('cividis').getLookupTable(0.0, 1.0, 256)   # MI/Score



        # GP Mean (col 0 + cbar at col 1)
        self.plot_mu = self.win_fields.addPlot(title="GP Mean", row=0, col=0)
        self._style_field(self.plot_mu)
        self.img_mu = pg.ImageItem()
        self.img_mu.setLookupTable(cm_mean)
        self.plot_mu.addItem(self.img_mu)
        self._setup_image(self.img_mu)
        self.cbar_mu = pg.ColorBarItem(values=self.levels_mu, colorMap=pg.colormap.get('turbo'))
        self.cbar_mu.setImageItem(self.img_mu)
        self.win_fields.addItem(self.cbar_mu, row=0, col=1)



        # Uncertainty (col 2 + cbar at col 3)
        self.plot_std = self.win_fields.addPlot(title="Uncertainty", row=0, col=2)
        self._style_field(self.plot_std)
        self.img_std = pg.ImageItem()
        self.img_std.setLookupTable(cm_uncert)
        self.plot_std.addItem(self.img_std)
        self._setup_image(self.img_std)
        self.cbar_std = pg.ColorBarItem(values=self.levels_std, colorMap=pg.colormap.get('inferno'))
        self.cbar_std.setImageItem(self.img_std)
        self.win_fields.addItem(self.cbar_std, row=0, col=3)



        # MI / Score (col 4 + cbar at col 5)
        self.plot_mi = self.win_fields.addPlot(title="MI", row=0, col=4)
        self._style_field(self.plot_mi)
        self.img_mi = pg.ImageItem()
        self.img_mi.setLookupTable(cm_mi)
        self.plot_mi.addItem(self.img_mi)
        self._setup_image(self.img_mi)
        self.cbar_mi = pg.ColorBarItem(values=self.levels_mi, colorMap=pg.colormap.get('cividis'))
        self.cbar_mi.setImageItem(self.img_mi)
        self.win_fields.addItem(self.cbar_mi, row=0, col=5)



        # Link all viewboxes so zoom/pan are identical

        self.plot_std.setXLink(self.plot_mu); self.plot_std.setYLink(self.plot_mu)

        self.plot_mi.setXLink(self.plot_mu);  self.plot_mi.setYLink(self.plot_mu)



        # Sample overlays

        self.samp_mu  = pg.ScatterPlotItem(size=12, pen='w', brush='w', symbol='x')

        self.samp_std = pg.ScatterPlotItem(size=12, pen='w', brush='w', symbol='x')

        self.samp_mi  = pg.ScatterPlotItem(size=12, pen='w', brush='w', symbol='x')

        self.plot_mu.addItem(self.samp_mu);   self.plot_std.addItem(self.samp_std); self.plot_mi.addItem(self.samp_mi)



        self.tgt_mu  = pg.ScatterPlotItem(size=16, pen='r', brush='r', symbol='star')

        self.tgt_std = pg.ScatterPlotItem(size=16, pen='r', brush='r', symbol='star')

        self.tgt_mi  = pg.ScatterPlotItem(size=16, pen='r', brush='r', symbol='star')

        self.plot_mu.addItem(self.tgt_mu);    self.plot_std.addItem(self.tgt_std);  self.plot_mi.addItem(self.tgt_mi)



        # ---- Window 2: Metrics ----

        self.win_metrics = pg.GraphicsLayoutWidget(title=f"Metrics - {title_suffix}")

        self.win_metrics.resize(1200, 800)

        self.win_metrics.show()



        self.plot_samples = self.win_metrics.addPlot(title="Samples", row=0, col=0)

        self._style_metric(self.plot_samples, 'Temperature [°C]')

        self.curve_samples = self.plot_samples.plot([], [], pen='b', symbol='o', symbolSize=6)



        self.plot_entropy = self.win_metrics.addPlot(title="Total Entropy", row=0, col=1)

        self._style_metric(self.plot_entropy, 'Entropy [nats]')

        self.curve_entropy = self.plot_entropy.plot([], [], pen='r', symbol='o', symbolSize=6)



        self.plot_mean_mi = self.win_metrics.addPlot(title="Mean MI", row=1, col=0)

        self._style_metric(self.plot_mean_mi, 'MI')

        self.curve_mean_mi = self.plot_mean_mi.plot([], [], pen='g', symbol='o', symbolSize=6)



        self.plot_hist = self.win_metrics.addPlot(title="Temperature Histogram", row=1, col=1)

        self.plot_hist.showGrid(x=True, y=True, alpha=0.3)

        self.plot_hist.setLabel('bottom', 'Temperature [°C]')

        self.plot_hist.setLabel('left', 'Frequency')

        self.hist_item = None



        # 30 Hz UI update (no frame saving)

        self.timer = QtCore.QTimer()

        self.timer.timeout.connect(self._on_tick)

        self.timer.start(int(1000 / 30))



    # ---- Public API -----------------------------------------------------



    def set_levels(self, mu=None, std=None, mi=None, mi_title="MI"):

        """Override colorbar/levels from the planner."""

        if mu is not None:  self.levels_mu  = (float(mu[0]),  float(mu[1]))

        if std is not None: self.levels_std = (float(std[0]), float(std[1]))

        if mi is not None:  self.levels_mi  = (float(mi[0]),  float(mi[1]))

        if _PG_AVAILABLE:

            self.cbar_mu.setLevels(self.levels_mu)

            self.cbar_std.setLevels(self.levels_std)

            self.cbar_mi.setLevels(self.levels_mi)

            self.plot_mi.setTitle(mi_title)



    def show(self):

        if not _PG_AVAILABLE:

            return

        self.win_fields.show()

        self.win_metrics.show()



    def push(self, mu, std, mi, X_obs, current_target, y_obs, H_hist, MI_hist):

        if not _PG_AVAILABLE:

            return

        with self._lock:

            self._latest = (

                mu.copy(), std.copy(), mi.copy(),

                X_obs.copy() if len(X_obs) else np.empty((0,2)),

                None if current_target is None else np.array(current_target, dtype=float),

                y_obs.copy(), np.array(H_hist, dtype=float), np.array(MI_hist, dtype=float)

            )



    def save_final_images(self, out_dir: Path):

        """Ask GUI thread to save final PNGs (no per-frame dumps)."""

        if not _PG_AVAILABLE:

            return

        self._save_path = out_dir

        self._save_requested = True

        # Let the GUI process a few frames so grabs aren't blank

        for _ in range(10):

            self.app.processEvents()

            QtCore.QThread.msleep(50)

    def save_figure(self, out_path: Path):

        """Save separate PNG files for fields and metrics windows."""

        if not _PG_AVAILABLE:

            return False

        try:

            # Process events to ensure latest render is visible

            for _ in range(10):

                self.app.processEvents()

                QtCore.QThread.msleep(50)

            # Grab both windows

            fields_img = self.win_fields.grab()

            metrics_img = self.win_metrics.grab()

            

            # Derive base path (remove extension and "_plots" suffix if present)

            base_name = out_path.stem

            if base_name.endswith("_plots"):

                base_name = base_name[:-6]  # Remove "_plots"

            elif base_name.endswith("_plot"):

                base_name = base_name[:-5]  # Remove "_plot"

            

            # Save separate images

            fields_path = out_path.parent / f"{base_name}_fields.png"

            metrics_path = out_path.parent / f"{base_name}_metrics.png"

            

            fields_img.save(str(fields_path), "PNG")

            metrics_img.save(str(metrics_path), "PNG")

            

            return True

        except Exception as e:

            print(f"Warning: Failed to save figures: {e}")

            return False



    def run(self):

        if not _PG_AVAILABLE:

            import time

            try:

                while True:

                    time.sleep(1.0)

            except KeyboardInterrupt:

                pass

            return

        self.app.exec_()



    # ---- Internals ------------------------------------------------------



    def _style_field(self, p):

        p.showGrid(x=True, y=True, alpha=0.3)

        p.setLabel('left', 'Y [m]')

        p.setLabel('bottom', 'X [m]')

        p.setLimits(xMin=0, xMax=self.W, yMin=0, yMax=self.H)

        p.setRange(xRange=[0, self.W], yRange=[0, self.H], padding=0.0)

        p.setAspectLocked(True, ratio=1.0)



    def _style_metric(self, p, ylab):

        p.showGrid(x=True, y=True, alpha=0.3)

        p.setLabel('bottom', 'Sample #')

        p.setLabel('left', ylab)



    def _setup_image(self, img):

        # Create a dummy image with the correct matrix shape

        dummy = np.zeros((int(self.ny), int(self.nx)), dtype=float)

        img.setImage(dummy, levels=(0.0, 1.0), autoLevels=False)

        img.resetTransform()



        # --- IMPORTANT: align pixel centers to the 1 m grid so every cell is the same size ---

        # We have nx,ny *nodes* from xs,ys (0..W and 0..H inclusive), so there are (nx-1)*(ny-1) cells.

        # For ImageItem (pixel centers), pad half a cell around the domain so each pixel spans exactly 1 m.

        cell_w = self.W / max(self.nx - 1, 1)

        cell_h = self.H / max(self.ny - 1, 1)



        # Extend the rect by half a cell on each side so pixel centers line up with integer coordinates

        rect = QtCore.QRectF(-0.5 * cell_w, -0.5 * cell_h, self.W + cell_w, self.H + cell_h)

        img.setRect(rect)



    def _update_image(self, img, data, levels):

        if data is None or data.size == 0:

            return

        d = np.array(data, dtype=float)

        d[~np.isfinite(d)] = 0.0

        # IMPORTANT: use fixed levels so color scale stays stable

        img.setImage(d, levels=levels, autoLevels=False)



    def _on_tick(self):

        if not _PG_AVAILABLE:

            return



        # Handle a final save request (ONLY two PNGs, no frame spam)

        if self._save_requested and self._save_path is not None:

            try:

                self._save_path.mkdir(parents=True, exist_ok=True)

                # Make sure latest render is on screen

                self.app.processEvents()

                QtCore.QThread.msleep(100)

                self.win_fields.grab().save(str(self._save_path / "final_fields.png"))

                self.win_metrics.grab().save(str(self._save_path / "final_metrics.png"))

                print(f"Saved final images to {self._save_path}")

            except Exception as e:

                print(f"Warning: Failed to save final images: {e}")

            finally:

                self._save_requested = False

                self._save_path = None



        with self._lock:

            data = self._latest

        if data is None:

            return



        mu, std, mi, X_obs, tgt, y_obs, H_hist, MI_hist = data

        if mu.size == 0 or mu.shape[0] != (self.nx * self.ny):

            return



        mu_img = mu.reshape(self.ny, self.nx).T

        std_img = std.reshape(self.ny, self.nx).T

        mi_img  = mi.reshape(self.ny, self.nx).T



        self._update_image(self.img_mu, mu_img,  self.levels_mu)

        self._update_image(self.img_std, std_img, self.levels_std)

        self._update_image(self.img_mi,  mi_img,  self.levels_mi)



        if len(X_obs):

            self.samp_mu.setData(X_obs[:,0], X_obs[:,1])

            self.samp_std.setData(X_obs[:,0], X_obs[:,1])

            self.samp_mi.setData(X_obs[:,0], X_obs[:,1])

        else:

            self.samp_mu.setData([], []); self.samp_std.setData([], []); self.samp_mi.setData([], [])



        if tgt is not None:

            self.tgt_mu.setData([tgt[0]], [tgt[1]])

            self.tgt_std.setData([tgt[0]], [tgt[1]])

            self.tgt_mi.setData([tgt[0]], [tgt[1]])

        else:

            self.tgt_mu.setData([], []); self.tgt_std.setData([], []); self.tgt_mi.setData([], [])



        # Metrics

        x  = np.arange(1, len(y_obs)+1, dtype=float)

        xe = np.arange(1, len(H_hist)+1, dtype=float)

        xm = np.arange(1, len(MI_hist)+1, dtype=float)

        self.curve_samples.setData(x, y_obs)

        self.curve_entropy.setData(xe, H_hist)

        self.curve_mean_mi.setData(xm, MI_hist)



        if self.hist_item is not None:

            self.plot_hist.removeItem(self.hist_item)

        if len(mu) > 0:

            y, edges = np.histogram(mu, bins=30)

            self.hist_item = pg.BarGraphItem(x=edges[:-1], height=y, width=np.diff(edges), brush='b')

            self.plot_hist.addItem(self.hist_item)
