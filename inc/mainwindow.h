#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QButtonGroup>

#include "slicezoomer.h"
#include "interval.h"
#include "def/def_coordinate.h"
#include "def/def_billon.h"

#include <qwt_plot_curve.h>
#include <qwt_plot_histogram.h>

namespace Ui
{
	class MainWindow;
}

class ContourCurve;
class Pith;
class PieChart;
class PlotSectorHistogram;
class PlotSliceHistogram;
class SectorHistogram;
class SliceHistogram;
class SliceView;
class KnotAreaHistogram;
class PlotKnotAreaHistogram;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow( QWidget *parent = 0 );
	~MainWindow();

	bool eventFilter( QObject *obj, QEvent *event );

private slots:
	void openDicom();
	void closeImage();
	void drawSlice();
	void setSlice( const int &sliceNumber );
	void setTypeOfView( const int &type );
	void updateSliceHistogram();
	void updatePith();
	void setMinimumOfSliceInterval( const int &min );
	void setMinimumOfSliceIntervalToCurrentSlice();
	void setMaximumOfSliceInterval( const int &max );
	void setMaximumOfSliceIntervalToCurrentSlice();
	void previousMaximumInSliceHistogram();
	void nextMaximumInSliceHistogram();
	void zoomInSliceView( const qreal &zoomFactor, const QPoint &focalPoint );
	void dragInSliceView( const QPoint &movementVector );
	void flowApplied();
	void setRestrictedAreaResolution( const int &resolution );
	void setRestrictedAreaThreshold( const int &threshold );
	void setRestrictedAreaBeginRadius( const int &radius );
	void setEdgeDetectionType( const int &type );
	void setCannyRadiusOfGaussianMask( const int &radius );
	void setCannySigmaOfGaussianMask( const double &sigma );
	void setCannyMinimumGradient( const int &minimumGradient );
	void setCannyMinimumDeviation( const double &minimumDeviation );
	void exportToDat();
	void exportToOfs();
	void exportToOfsAll();
	void exportToOfsRestricted();
	void exportSectorToOfs();
	void exportAllSectorInAllIntervalsToOfs();
	void exportToV3D();
	void exportFlowToV3D();
	void exportHistogramToSep();
	void exportMovementsToV3D();
	void selectSliceInterval( const int &index );
	void selectCurrentSliceInterval();
	void selectSectorInterval( const int &index );
	void selectCurrentSectorInterval();
	void exportSectorToPgm3D();
	void exportConnexComponentToPgm3D();
	void exportSliceHistogram();
	void exportSectorDiagramAndHistogram();
	void exportKnotIntervalHistogram();
	void exportContours();
	void exportContourComponentToPgm3D();
		void createVoxelSetAllIntervals(std::vector<iCoord3D> &vectVoxels, bool useOldMethod=false);
		void createVoxelSet(std::vector<iCoord3D> &vectVoxels);
		void exportAllContourComponentOfVoxels();
		void exportAllContourComponentOfVoxelsAllIntervals();
		void exportAllContourComponentOfVoxelsAllIntervalsOldMethod();


private:
	void openNewBillon(const QString &folderName = "");
	void initComponentsValues();
	void updateUiComponentsValues();
	void enabledComponents();
	void updateSectorHistogram( const Interval<uint> &interval );

private:
	Ui::MainWindow *_ui;

	Billon *_billon;
	Billon *_sectorBillon;
	Billon *_componentBillon;

	QImage _mainPix;
	SliceZoomer _sliceZoomer;

	SliceView *_sliceView;
	ContourCurve *_contourCurve;

	SliceHistogram *_sliceHistogram;
	PlotSliceHistogram *_plotSliceHistogram;

	PieChart *_pieChart;
	SectorHistogram *_sectorHistogram;
	PlotSectorHistogram * _plotSectorHistogram;

	KnotAreaHistogram *_knotAreaHistogram;
	PlotKnotAreaHistogram *_plotKnotAreaHistogram;

	uint _currentSlice;
	int _currentMaximum;
	uint _currentSector;
};

#endif // MAINWINDOW_H
