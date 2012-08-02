#ifndef BILLON_H
#define BILLON_H

#include "global.h"
#include "marrow.h"

#include <QDebug>
#include <armadillo>

#include <QPolygon>

template< typename T >
class BillonTpl : public arma::Cube<T>
{
public:
	BillonTpl();
	BillonTpl( const int &width, const int &height, const int &depth );
	BillonTpl( const BillonTpl &billon );

	T minValue() const;
	T maxValue() const;
	qreal voxelWidth() const;
	qreal voxelHeight() const;
	qreal voxelDepth() const;

	void setMinValue( const T &value );
	void setMaxValue( const T &value );
	void setVoxelSize( const qreal &width, const qreal &height, const qreal &depth );

	QVector<rCoord2D> getAllRestrictedAreaVertex( const int &nbPolygonsPoints, const int &threshold, const Marrow *marrow = 0 ) const;
	qreal getRestrictedAreaBoudingBoxRadius( const Marrow *marrow, const int &nbPolygonPoints, int intensityThreshold ) const;
	qreal getRestrictedAreaMeansRadius( const Marrow *marrow, const int &nbPolygonPoints, int intensityThreshold ) const;

	QList<rCoord2D> extractEdges( const Marrow *marrow, const int &sliceNumber, const int &componentNumber );

protected:
	T _minValue;
	T _maxValue;
	qreal _voxelWidth;	// Largeur d'un voxel en cm
	qreal _voxelHeight; // Hauteur d'un voxel en cm
	qreal _voxelDepth;	// Profondeur d'un voxel en cm
};

template< typename T >
BillonTpl<T>::BillonTpl() : arma::Cube<T>(), _minValue(static_cast<T>(0)), _maxValue(static_cast<T>(0)), _voxelWidth(0.), _voxelHeight(0.), _voxelDepth(0.) {}

template< typename T >
BillonTpl<T>::BillonTpl( const int &width, const int &height, const int &depth ) : arma::Cube<T>(height,width,depth), _minValue(static_cast<T>(0)), _maxValue(static_cast<T>(0)), _voxelWidth(0.), _voxelHeight(0.), _voxelDepth(0.) {}

template< typename T >
BillonTpl<T>::BillonTpl( const BillonTpl &billon ) : arma::Cube<T>(billon), _minValue(billon._minValue), _maxValue(billon._maxValue), _voxelWidth(billon._voxelWidth), _voxelHeight(billon._voxelHeight), _voxelDepth(billon._voxelDepth) {}

template< typename T >
T BillonTpl<T>::minValue() const {
	return _minValue;
}

template< typename T >
T BillonTpl<T>::maxValue() const {
	return _maxValue;
}

template< typename T >
qreal BillonTpl<T>::voxelWidth() const {
	return _voxelWidth;
}

template< typename T >
qreal BillonTpl<T>::voxelHeight() const {
	return _voxelHeight;
}

template< typename T >
qreal BillonTpl<T>::voxelDepth() const {
	return _voxelDepth;
}

template< typename T >
void BillonTpl<T>::setMinValue( const T &value ) {
	_minValue = value;
}

template< typename T >
void BillonTpl<T>::setMaxValue( const T &value ) {
	_maxValue = value;
}

template< typename T >
void BillonTpl<T>::setVoxelSize(const qreal &width, const qreal &height, const qreal &depth) {
	_voxelWidth = width;
	_voxelHeight = height;
	_voxelDepth = depth;
}

template< typename T >
QVector<rCoord2D> BillonTpl<T>::getAllRestrictedAreaVertex( const int &nbPolygonPoints, const int &threshold, const Marrow *marrow ) const {
	 QVector<rCoord2D> vectAllVertex;
	 const int nbSlices = this->n_slices;
	 const int thresholdRestrict = threshold-1;
	 for ( int indexSlice = 0 ; indexSlice<nbSlices ; ++indexSlice ) {
		 const arma::Mat<T> &currentSlice = this->slice(indexSlice);

		 const int sliceWidth = currentSlice.n_cols;
		 const int sliceHeight = currentSlice.n_rows;
		 const int xCenter = (marrow != 0 && marrow->interval().containsClosed(indexSlice))?marrow->at(indexSlice).x:sliceWidth/2;
		 const int yCenter = (marrow != 0 && marrow->interval().containsClosed(indexSlice))?marrow->at(indexSlice).y:sliceHeight/2;

		 qreal xEdge, yEdge, orientation, cosAngle, sinAngle;
		 orientation = 0.;
		 for ( int i=0 ; i<nbPolygonPoints ; ++i ) {
			 orientation += (TWO_PI/static_cast<qreal>(nbPolygonPoints));
			 cosAngle = qCos(orientation);
			 sinAngle = -qSin(orientation);
			 xEdge = xCenter + 5*cosAngle;
			 yEdge = yCenter + 5*sinAngle;
			 while ( xEdge>0. && yEdge>0. && xEdge<sliceWidth && yEdge<sliceHeight && currentSlice.at(yEdge,xEdge) > thresholdRestrict ) {
					 xEdge += cosAngle;
					 yEdge += sinAngle;
			 }
			 vectAllVertex.push_back(rCoord2D(xEdge,yEdge));
		 }
	 }
	return vectAllVertex;
}

template< typename T >
qreal BillonTpl<T>::getRestrictedAreaBoudingBoxRadius( const Marrow *marrow, const int &nbPolygonPoints, int intensityThreshold ) const
{
	QPolygon polygon(nbPolygonPoints);
	int polygonPoints[2*nbPolygonPoints+2];
	qreal xEdge, yEdge, orientation, cosAngle, sinAngle, radius;
	int i,k,counter;

	const int width = this->n_cols;
	const int height = this->n_rows;
	const int depth = this->n_slices;

	radius = 0.;
	for ( k=0 ; k<depth ; ++k ) {
		const arma::Mat<T> &currentSlice = this->slice(k);
		const int xCenter = (marrow != 0 && marrow->interval().containsClosed(k))?marrow->at(k-marrow->interval().minValue()).x:width/2;
		const int yCenter = (marrow != 0 && marrow->interval().containsClosed(k))?marrow->at(k-marrow->interval().minValue()).y:height/2;

		orientation = 0.;
		counter = 0;
		for ( i=0 ; i<nbPolygonPoints ; ++i )
		{
			orientation += (TWO_PI/static_cast<qreal>(nbPolygonPoints));
			cosAngle = qCos(orientation);
			sinAngle = -qSin(orientation);
			xEdge = xCenter + 10*cosAngle;
			yEdge = yCenter + 10*sinAngle;
			while ( xEdge>0 && yEdge>0 && xEdge<width && yEdge<height && currentSlice.at(yEdge,xEdge) > intensityThreshold )
			{
				xEdge += cosAngle;
				yEdge += sinAngle;
			}
			polygonPoints[counter++] = xEdge;
			polygonPoints[counter++] = yEdge;
		}
		polygonPoints[counter++] = polygonPoints[0];
		polygonPoints[counter] = polygonPoints[1];

		polygon.setPoints(nbPolygonPoints+1,polygonPoints);

		radius += 0.5*qSqrt(polygon.boundingRect().width()*polygon.boundingRect().width() + polygon.boundingRect().height()*polygon.boundingRect().height());
	}

	radius/=depth;
	qDebug() << "Rayon de la boite englobante : " << radius << " (" << radius*_voxelWidth << " mm)";
	return radius*_voxelWidth;
}

template< typename T >
qreal BillonTpl<T>::getRestrictedAreaMeansRadius( const Marrow *marrow, const int &nbPolygonPoints, int intensityThreshold ) const
{
	qreal xEdge, yEdge, orientation, cosAngle, sinAngle, radius;
	int i,k;

	const int width = this->n_cols;
	const int height = this->n_rows;
	const int depth = this->n_slices;

	radius = 0.;
	for ( k=0 ; k<depth ; ++k ) {
		const arma::Mat<T> &currentSlice = this->slice(k);
		const int xCenter = (marrow != 0 && marrow->interval().containsClosed(k))?marrow->at(k-marrow->interval().minValue()).x:width/2;
		const int yCenter = (marrow != 0 && marrow->interval().containsClosed(k))?marrow->at(k-marrow->interval().minValue()).y:height/2;

		orientation = 0.;
		for ( i=0 ; i<nbPolygonPoints ; ++i )
		{
			orientation += (TWO_PI/static_cast<qreal>(nbPolygonPoints));
			cosAngle = qCos(orientation);
			sinAngle = -qSin(orientation);
			xEdge = xCenter + 10*cosAngle;
			yEdge = yCenter + 10*sinAngle;
			while ( xEdge>0 && yEdge>0 && xEdge<width && yEdge<height && currentSlice.at(yEdge,xEdge) > intensityThreshold )
			{
				xEdge += cosAngle;
				yEdge += sinAngle;
			}
			xEdge -= xCenter;
			yEdge -= yCenter;
			radius += qSqrt( xEdge*xEdge + yEdge*yEdge );
		}
	}

	radius/=(depth*nbPolygonPoints);
	qDebug() << "Rayon de la boite englobante (en pixels) : " << radius;
	return radius;
}


template< typename T >
QList<rCoord2D> BillonTpl<T>::extractEdges( const Marrow *marrow, const int &sliceNumber, const int &componentNumber ) {
	// Find the pixel closest to the pith
	const arma::Mat<T> &currentSlice = this->slice(sliceNumber);
	const int width = this->n_cols;
	const int height = this->n_rows;
	const int xCenter = marrow != 0 ? marrow->at(sliceNumber).x : width/2;
	const int yCenter = marrow != 0 ? marrow->at(sliceNumber).y : height/2;

	qreal radius, radiusMax, orientation, xEdge, yEdge, cosAngle, sinAngle;
	int step;
	bool edgeFind = false;
	radius = 1;
	radiusMax = qMin( qMin(xCenter,width-xCenter), qMin(yCenter,height-yCenter) );

	step = 0;
	while ( !edgeFind && radius < radiusMax )
	{
		step++;
		for ( orientation = 0. ; orientation <= TWO_PI && !edgeFind ; orientation+=(PI/180.) )
		{
			cosAngle = qCos(orientation);
			sinAngle = -qSin(orientation);
			xEdge = xCenter + step*cosAngle;
			yEdge = yCenter + step*sinAngle;
			radius = qMax(radius,qSqrt( (xCenter-xEdge)*(xCenter-xEdge) + (yCenter-yEdge)*(yCenter-yEdge) ));
			if ( xEdge>0 && yEdge>0 && xEdge<width && yEdge<height && currentSlice.at(yEdge,xEdge) == componentNumber ) edgeFind = true;
		}
	}

	rCoord2D position;
	if ( edgeFind ) {
		qDebug() << "Pixel le plus proche de la moelle : ( " << xEdge << ", " << yEdge << " )";
	}
	else {
		qDebug() << "Aucun pixel et donc aucune composante connexe";
		xEdge = 0;
		yEdge = 0;
	}
	position.x = xEdge;
	position.y = yEdge;

	// Suivi du contour
	QList<rCoord2D> contourPoints;
	if ( edgeFind ) {
		int xBegin, yBegin, xCurrent, yCurrent, interdit, j;
		QVector<int> vx(8), vy(8);

		xBegin = xCurrent = xEdge;
		yBegin = yCurrent = yEdge;
		interdit = orientation*8./TWO_PI;
		interdit = (interdit+4)%8;
		do
		{
			contourPoints.append(rCoord2D(xCurrent,yCurrent));
			vx[0] = vx[1] = vx[7] = xCurrent+1;
			vx[2] = vx[6] = xCurrent;
			vx[3] = vx[4] = vx[5] = xCurrent-1;
			vy[1] = vy[2] = vy[3] = yCurrent-1;
			vy[0] = vy[4] = yCurrent;
			vy[5] = vy[6] = vy[7] = yCurrent+1;
			j = (interdit+1)%8;
			while ( currentSlice.at(vy[j%8],vx[j%8]) != componentNumber && j < interdit+8 ) ++j;
			xCurrent = vx[j%8];
			yCurrent = vy[j%8];
			interdit = (j+4)%8;
		}
		while ( xBegin != xCurrent || yBegin != yCurrent );
	}
	else {
		contourPoints.append(position);
	}

	return contourPoints;
}


#endif // BILLON_H
