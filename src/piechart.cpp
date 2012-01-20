#include "inc/piechart.h"

#include "inc/pie_def.h"
#include "inc/piepart.h"

#include <QPainter>

#include <cmath>
#include <limits>

#define DOUBLE_ERR_POS 0.0000000001
#define DOUBLE_ERR_NEG (-DOUBLE_ERR_POS)

namespace {
	bool IS_EQUAL( const double & v1, const double &v2 ) {
		const double diff = v1 - v2;
		return diff < DOUBLE_ERR_POS && diff > DOUBLE_ERR_NEG;
	}
}

PieChart::PieChart( const double &orientation, const int &nbSectors ) : _orientation(orientation), _angle(TWO_PI/static_cast<double>(nbSectors)) {
	updateSectors();
}

/*******************************
 * Public getters
 *******************************/

double PieChart::orientation() const {
	return _orientation;
}

double PieChart::angle() const {
	return _angle;
}

int PieChart::nbSectors() const {
	return (TWO_PI/_angle);
}

int PieChart::partOfAngle( const double &angle ) const {
	int sectorId;
	bool ok = false;
	const int nbSectors = _sectors.size();
	for ( sectorId = 0 ; !ok && (sectorId < nbSectors) ; sectorId += !ok ) {
		ok = _sectors.at(sectorId).contains(angle);
	}
	return (sectorId<nbSectors)?sectorId:0;
}

/*******************************
 * Public setters
 *******************************/

void PieChart::setOrientation( const double &orientation ) {
	_orientation = orientation;
	updateSectors();
}

void PieChart::setSectorsNumber( const int &nbSectors ) {
	_angle = TWO_PI/static_cast<double>(nbSectors);
	updateSectors();
}

void PieChart::draw( QPainter &painter, const int &sectorIdx, const Coord2D &center ) const {
	const int width = painter.window().width();
	const int height = painter.window().height();
	const int centerX = center.x;
	const int centerY = center.y;

	// Liste qui va contenir les angles des deux côté du secteur à dessiner
	// Permet de factoriser le code de calcul des coordonnées juste en dessous
	QList<double> twoSides;
	twoSides.append( TWO_PI-_sectors.at(sectorIdx).rightAngle() );
	twoSides.append( TWO_PI-_sectors.at(sectorIdx).leftAngle() );

	painter.setPen(QColor(0,255,0));

	// Dessin des deux côtés du secteur
	double angle, x1,y1,x2,y2;
	while ( !twoSides.isEmpty() ) {
		// Calcul des coordonnées du segment à tracer
		angle = twoSides.takeLast();
		x1 = x2 = centerX;
		y1 = y2 = centerY;
		if ( IS_EQUAL(angle,PI_ON_TWO) ) y2 = height;
		else if ( IS_EQUAL(angle,THREE_PI_ON_TWO) ) y1 = 0;
		else {
			const double a = tan(angle);
			const double b = centerY - (a*centerX);
			if ( angle < PI_ON_TWO || angle > THREE_PI_ON_TWO ) {
				x2 = width;
				y2 = a*width+b;
			}
			else {
				x1 = 0;
				y1 = b;
			}
		}

		// Tracé du segment droit
		painter.drawLine(x1,y1,x2,y2);
	}
}

/*******************************
 * Private setters
 *******************************/

void PieChart::updateSectors() {
	_sectors.clear();
	const int nbSectors = TWO_PI/_angle;
	double currentOrientation = _orientation;
	for ( int i=0 ; i<nbSectors ; ++i ) {
		_sectors.append(PiePart( fmod( currentOrientation, TWO_PI ), _angle ));
		currentOrientation += _angle;
	}
}
