#ifndef IMAGE3D_H
#define IMAGE3D_H

#include <armadillo>

using namespace arma;

class Billon : public icube
{
public:
	Billon();
	Billon(int,int,int);

	int minValue() const;
	int maxValue() const;

	void setMinValue( const int );
	void setMaxValue( const int );

private:
	int _minValue;
	int _maxValue;
};

inline
int Billon::minValue() const {
	return _minValue;
}

inline
int Billon::maxValue() const {
	return _maxValue;
}

inline
void Billon::setMinValue( const int max ) {
	_minValue = max;
}

inline
void Billon::setMaxValue( const int min ) {
	_maxValue = min;
}

#endif // IMAGE3D_H
