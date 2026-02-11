"""
Production-Grade Scientific Plotting Engine
============================================

A matplotlib-based plotting system designed for academic research,
numerical analysis, and publication-quality visualization.

Features:
- Strict engine-style naming conventions
- Research-grade dark theme
- Multi-panel support with automatic layout
- Animation capabilities
- Scientific annotation tools
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Callable, Dict, List, Tuple, Any

class PlotEngine:
    """
    Core plotting engine for single-panel scientific figures.

    Provides publication-quality rendering with a clean dark theme,
    proper typography, and scientific annotation capabilities.
    """

    # Static color scheme for research dark mode
    s_colors = {
        'primary': '#5BC0DE',  # Soft cyan
        'secondary': '#F0E68C',  # Pale yellow
        'tertiary': '#90C695',  # Muted green
        'figure_bg': '#0f1116',  # Deep dark background (from user theme)
        'axes_bg': '#0f1116',  # Same as figure background
        'grid': '#FFFFFF',  # White grid (with alpha)
        'spine': '#FFFFFF',  # White edges
        'text': '#FFFFFF',  # White text
        'tick': '#FFFFFF',  # White ticks
    }

    # Typography configuration
    s_fontSizes = {
        'suptitle': 14,
        'title': 11,
        'label': 10,
        'tick': 8,
        'legend': 8,
        'annotation': 8,
    }

    def __init__(self, figsize: Tuple[float, float] = (10, 6)):
        """
        Initialize the plot engine.

        Args:
            figsize: Figure dimensions in inches (width, height)
        """
        self.m_figure = None
        self.m_axes = None
        self.m_figsize = figsize
        self.m_lineCounter = 0

        self.internalInitializeFigure()
        self.internalApplyTheme()

    def internalInitializeFigure(self) -> None:
        """Create the matplotlib figure and axes."""
        self.m_figure, self.m_axes = plt.subplots(figsize=self.m_figsize, facecolor=self.s_colors['figure_bg'])
        self.m_figure.patch.set_facecolor(self.s_colors['figure_bg'])
        self.m_axes.set_facecolor(self.s_colors['axes_bg'])
        self.m_axes.patch.set_facecolor(self.s_colors['axes_bg'])

    def internalApplyTheme(self) -> None:
        """Apply research-grade dark theme to the axes."""
        # Configure spines
        for spine in self.m_axes.spines.values():
            spine.set_edgecolor(self.s_colors['spine'])
            spine.set_linewidth(0.8)

        # Configure grid
        self.m_axes.grid(True, which='major', linestyle='-',
                         linewidth=0.5, alpha=0.2, color=self.s_colors['grid'])

        # Configure ticks
        self.m_axes.tick_params(
            axis='both',
            which='major',
            labelsize=self.s_fontSizes['tick'],
            colors=self.s_colors['tick'],
            length=4,
            width=0.8,
            direction='out'
        )

        # Set text colors
        self.m_axes.xaxis.label.set_color(self.s_colors['text'])
        self.m_axes.yaxis.label.set_color(self.s_colors['text'])

    def addLine(self, xData: np.ndarray, yData: np.ndarray,
                label: Optional[str] = None,
                color: Optional[str] = None,
                linewidth: float = 1.5,
                linestyle: str = '-',
                alpha: float = 1.0) -> None:
        """
        Add a line plot to the axes.

        Args:
            xData: X-axis data points
            yData: Y-axis data points
            label: Line label for legend
            color: Line color (defaults to color cycle)
            linewidth: Line width in points
            linestyle: Line style ('-', '--', '-.', ':')
            alpha: Line transparency (0.0 to 1.0)
        """
        if color is None:
            colorCycle = [self.s_colors['primary'],
                          self.s_colors['secondary'],
                          self.s_colors['tertiary']]
            color = colorCycle[self.m_lineCounter % len(colorCycle)]
            self.m_lineCounter += 1

        self.m_axes.plot(xData, yData, label=label, color=color,
                         linewidth=linewidth, linestyle=linestyle, alpha=alpha)

    def addScatter(self, xData: np.ndarray, yData: np.ndarray,
                   label: Optional[str] = None,
                   color: Optional[str] = None,
                   marker: str = 'o',
                   size: float = 30,
                   alpha: float = 0.7) -> None:
        """
        Add a scatter plot to the axes.

        Args:
            xData: X-axis data points
            yData: Y-axis data points
            label: Scatter label for legend
            color: Marker color
            marker: Marker style
            size: Marker size
            alpha: Marker transparency
        """
        if color is None:
            color = self.s_colors['primary']

        self.m_axes.scatter(xData, yData, label=label, color=color,
                            marker=marker, s=size, alpha=alpha, edgecolors='none')

    def addHorizontalLine(self, yValue: float,
                          color: Optional[str] = None,
                          linestyle: str = '--',
                          linewidth: float = 1.0,
                          alpha: float = 0.5,
                          label: Optional[str] = None) -> None:
        """Add a horizontal reference line."""
        if color is None:
            color = self.s_colors['grid']

        self.m_axes.axhline(y=yValue, color=color, linestyle=linestyle,
                            linewidth=linewidth, alpha=alpha, label=label)

    def addVerticalLine(self, xValue: float,
                        color: Optional[str] = None,
                        linestyle: str = '--',
                        linewidth: float = 1.0,
                        alpha: float = 0.5,
                        label: Optional[str] = None) -> None:
        """Add a vertical reference line."""
        if color is None:
            color = self.s_colors['grid']

        self.m_axes.axvline(x=xValue, color=color, linestyle=linestyle,
                            linewidth=linewidth, alpha=alpha, label=label)

    def setTitle(self, title: str) -> None:
        """
        Set the axes title with proper formatting.

        Args:
            title: Title text (supports LaTeX mathtext)
        """
        self.m_axes.set_title(title, fontsize=self.s_fontSizes['title'],
                              color=self.s_colors['text'], pad=10)

    def setLabels(self, xlabel: str, ylabel: str) -> None:
        """
        Set axis labels with proper formatting.

        Args:
            xlabel: X-axis label (supports LaTeX mathtext)
            ylabel: Y-axis label (supports LaTeX mathtext)
        """
        self.m_axes.set_xlabel(xlabel, fontsize=self.s_fontSizes['label'])
        self.m_axes.set_ylabel(ylabel, fontsize=self.s_fontSizes['label'])

    def setLimits(self, xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None) -> None:
        """
        Set axis limits.

        Args:
            xlim: X-axis limits (min, max)
            ylim: Y-axis limits (min, max)
        """
        if xlim is not None:
            self.m_axes.set_xlim(xlim)
        if ylim is not None:
            self.m_axes.set_ylim(ylim)

    def addLegend(self, location: str = 'best',
                  framealpha: float = 0.7,
                  edgecolor: Optional[str] = None) -> None:
        """
        Add a minimal, clean legend.

        Args:
            location: Legend location
            framealpha: Frame transparency
            edgecolor: Frame edge color
        """
        if edgecolor is None:
            edgecolor = self.s_colors['spine']

        legend = self.m_axes.legend(
            loc=location,
            fontsize=self.s_fontSizes['legend'],
            framealpha=framealpha,
            edgecolor=edgecolor,
            fancybox=False,
            shadow=False
        )

        # Style the legend
        legend.get_frame().set_facecolor(self.s_colors['axes_bg'])
        for text in legend.get_texts():
            text.set_color(self.s_colors['text'])

    def annotateInlineText(self, xPos: float, yPos: float, text: str,
                           horizontalAlign: str = 'left',
                           verticalAlign: str = 'bottom',
                           fontsize: Optional[int] = None,
                           color: Optional[str] = None) -> None:
        """
        Add inline text annotation.

        Args:
            xPos: X position in data coordinates
            yPos: Y position in data coordinates
            text: Annotation text (supports LaTeX mathtext)
            horizontalAlign: Horizontal alignment
            verticalAlign: Vertical alignment
            fontsize: Font size (defaults to annotation size)
            color: Text color
        """
        if fontsize is None:
            fontsize = self.s_fontSizes['annotation']
        if color is None:
            color = self.s_colors['text']

        self.m_axes.text(xPos, yPos, text,
                         fontsize=fontsize,
                         color=color,
                         ha=horizontalAlign,
                         va=verticalAlign)

    def annotateMetricsBlock(self, metrics: Dict[str, Any],
                             position: str = 'upper right',
                             fontSize: Optional[int] = None) -> None:
        """
        Add a small metrics block annotation.

        Displays key-value pairs in a clean, minimal format.
        Ideal for showing error metrics, norms, or statistics.

        Args:
            metrics: Dictionary of metric names and values
            position: Position ('upper right', 'upper left', etc.)
            fontSize: Font size for metrics
        """
        if fontSize is None:
            fontSize = self.s_fontSizes['annotation']

        # Build metrics text
        metricsText = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])

        # Position mapping
        positionMap = {
            'upper right': (0.97, 0.97, 'right', 'top'),
            'upper left': (0.03, 0.97, 'left', 'top'),
            'lower right': (0.97, 0.03, 'right', 'bottom'),
            'lower left': (0.03, 0.03, 'left', 'bottom'),
        }

        xPos, yPos, ha, va = positionMap.get(position, positionMap['upper right'])

        # Add text box
        self.m_axes.text(xPos, yPos, metricsText,
                         transform=self.m_axes.transAxes,
                         fontsize=fontSize,
                         color=self.s_colors['text'],
                         ha=ha, va=va,
                         bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor=self.s_colors['axes_bg'],
                                   edgecolor=self.s_colors['spine'],
                                   alpha=0.8,
                                   linewidth=0.5))

    def saveFigure(self, filepath: str, dpi: int = 300,
                   transparentBackground: bool = False) -> None:
        """
        Save the figure to disk.

        Args:
            filepath: Output file path
            dpi: Resolution in dots per inch (300+ for publication)
            transparentBackground: Use transparent background
        """
        self.m_figure.tight_layout()
        self.m_figure.savefig(filepath, dpi=dpi,
                              facecolor=self.m_figure.get_facecolor(),
                              transparent=transparentBackground,
                              bbox_inches='tight')

    def show(self) -> None:
        """Display the figure."""
        self.m_figure.tight_layout()
        plt.show()

    def clear(self) -> None:
        """Clear the axes content."""
        self.m_axes.clear()
        self.internalApplyTheme()
        self.m_lineCounter = 0

    def close(self) -> None:
        """Close the figure and release resources."""
        if self.m_figure is not None:
            plt.close(self.m_figure)
            self.m_figure = None
            self.m_axes = None


class MultiPanelEngine:
    """
    Multi-panel plotting engine for vertically stacked subplots.

    Designed for dense research figures with 5-10 panels,
    automatic spacing, optional shared axes, and consistent theming.
    """

    def __init__(self, nrows: int, figsize: Tuple[float, float] = (10, 12),
                 sharex: bool = False):
        """
        Initialize multi-panel engine.

        Args:
            nrows: Number of vertically stacked panels
            figsize: Figure dimensions in inches (width, height)
            sharex: Share X-axis across all panels
        """
        self.m_nrows = nrows
        self.m_figsize = figsize
        self.m_sharex = sharex
        self.m_figure = None
        self.m_axes = []
        self.m_panels = []

        self.internalInitializePanels()

    def internalInitializePanels(self) -> None:
        """Create figure with vertically stacked panels."""
        # Create subplots with optimized spacing
        self.m_figure, axesArray = plt.subplots(
            self.m_nrows, 1,
            figsize=self.m_figsize,
            sharex=self.m_sharex,
            facecolor=PlotEngine.s_colors['figure_bg']
        )

        # Handle single vs multiple axes
        if self.m_nrows == 1:
            self.m_axes = [axesArray]
        else:
            self.m_axes = axesArray

        # Set figure background
        self.m_figure.patch.set_facecolor(PlotEngine.s_colors['figure_bg'])

        # Create panel engines
        for ax in self.m_axes:
            # Set axes background to dark
            ax.set_facecolor(PlotEngine.s_colors['axes_bg'])
            ax.patch.set_facecolor(PlotEngine.s_colors['axes_bg'])

            panel = self.internalCreatePanelFromAxes(ax)
            self.m_panels.append(panel)

        # Adjust spacing for dense layouts
        self.m_figure.subplots_adjust(hspace=0.3)

    def internalCreatePanelFromAxes(self, ax) -> PlotEngine:
        """
        Create a PlotEngine instance from an existing axes.

        Args:
            ax: Matplotlib axes object

        Returns:
            PlotEngine instance wrapping the axes
        """
        panel = PlotEngine.__new__(PlotEngine)
        panel.m_figure = self.m_figure
        panel.m_axes = ax
        panel.m_figsize = self.m_figsize
        panel.m_lineCounter = 0

        # Apply theme to the panel
        panel.internalApplyTheme()

        return panel

    def getPanel(self, index: int) -> PlotEngine:
        """
        Get a specific panel for plotting.

        Args:
            index: Panel index (0-based)

        Returns:
            PlotEngine instance for the requested panel
        """
        if index < 0 or index >= self.m_nrows:
            raise IndexError(f"Panel index {index} out of range [0, {self.m_nrows - 1}]")

        return self.m_panels[index]

    def setMainTitle(self, title: str, fontSize: Optional[int] = None) -> None:
        """
        Set the main figure title (suptitle).

        Args:
            title: Main title text (supports LaTeX mathtext)
            fontSize: Font size (defaults to suptitle size)
        """
        if fontSize is None:
            fontSize = PlotEngine.s_fontSizes['suptitle']

        self.m_figure.suptitle(title, fontsize=fontSize,
                               color=PlotEngine.s_colors['text'],
                               y=0.995)

    def saveFigure(self, filepath: str, dpi: int = 300,
                   transparentBackground: bool = False) -> None:
        """
        Save the multi-panel figure to disk.

        Args:
            filepath: Output file path
            dpi: Resolution in dots per inch
            transparentBackground: Use transparent background
        """
        self.m_figure.tight_layout(rect=[0, 0, 1, 0.99])
        self.m_figure.savefig(filepath, dpi=dpi,
                              facecolor=self.m_figure.get_facecolor(),
                              transparent=transparentBackground,
                              bbox_inches='tight')

    def show(self) -> None:
        """Display the multi-panel figure."""
        self.m_figure.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()

    def close(self) -> None:
        """Close the figure and release resources."""
        if self.m_figure is not None:
            plt.close(self.m_figure)
            self.m_figure = None
            self.m_axes = []
            self.m_panels = []


class AnimationEngine:
    """
    Animation engine for deterministic scientific animations.

    Supports MP4 and GIF export with consistent theming
    and flicker-free rendering using blitting.
    """

    def __init__(self, plotEngine: PlotEngine):
        """
        Initialize animation engine.

        Args:
            plotEngine: PlotEngine instance to animate
        """
        self.m_plotEngine = plotEngine
        self.m_animation = None

    def animate(self, updateFn: Callable[[int], List],
                frames: int,
                interval: int = 50,
                useBlit: bool = True) -> None:
        """
        Create an animation.

        Args:
            updateFn: Update function that takes frame number and returns artists
            frames: Number of frames
            interval: Delay between frames in milliseconds
            useBlit: Use blitting for performance
        """
        self.m_animation = animation.FuncAnimation(
            self.m_plotEngine.m_figure,
            updateFn,
            frames=frames,
            interval=interval,
            blit=useBlit,
            repeat=True
        )

    def saveAnimation(self, filepath: str, fps: int = 30,
                      codec: str = 'h264') -> None:
        """
        Save animation to file.

        Args:
            filepath: Output file path (.mp4 or .gif)
            fps: Frames per second
            codec: Video codec ('h264' for MP4)
        """
        if self.m_animation is None:
            raise RuntimeError("No animation created. Call animate() first.")

        if filepath.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, codec=codec,
                                            bitrate=1800)

        self.m_animation.save(filepath, writer=writer, dpi=100)

    def show(self) -> None:
        """Display the animation."""
        if self.m_animation is None:
            raise RuntimeError("No animation created. Call animate() first.")

        plt.show()


# Configure matplotlib for research-grade rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.default'] = 'regular'