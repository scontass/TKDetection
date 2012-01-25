#ifndef MARROW_H
#define MARROW_H

#include <QList>
#include "marrow_def.h"

class QPainter;

class Marrow : public QList<Coord2D>
{
public:
	Marrow();
	Marrow( const int &begin, const int &end );

	int beginSlice() const;
	int endSlice() const;

	void draw( QPainter &painter, const int &sliceIdx ) const;

private:
	int _begin;
	int _end;
};

#endif // MARROW_H