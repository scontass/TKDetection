#ifndef BILLONALGORITHMS_H
#define BILLONALGORITHMS_H

#include "def/def_billon.h"
#include "def/def_coordinate.h"
#include "inc/coordinate.h"

template <typename T> class Interval;
template <typename T> class QVector;
class Pith;

namespace BillonAlgorithms
{
	iCoord2D findNearestPointOfThePith( const Slice &slice, const iCoord2D & sliceCenter, const int &intensityThreshold );
	QVector<iCoord2D> extractContour( const Slice &slice, const iCoord2D & sliceCenter, const int &intensityThreshold, iCoord2D startPoint = iCoord2D(-1,-1) );
	qreal restrictedAreaMeansRadius( const Billon &billon, const uint &nbPolygonPoints, const int &intensityThreshold );
	QVector<rCoord2D> restrictedAreaVertex(const Billon &billon, const Interval<uint> &sliceInterval, const uint &nbPolygonVertex, const int &intensityThreshold );
}

#endif // BILLONALGORITHMS_H