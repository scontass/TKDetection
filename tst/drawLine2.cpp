void PithExtractorBoukadida::drawLine2(arma::Mat<int> &slice, const iCoord2D &origin, const qreal &orientation ) const
{
	const int heightMinusOne = slice.n_rows-1;
	const int widthMinusOne = slice.n_cols-1;
	const int originX = origin.x;
	const int originY = origin.y;
	const qreal orientationInv = 1./orientation;

	qreal x, y;

  int i;
  int dcol, dlig, dmin;

	if ( orientation >= 1. )
    {
      x = originX;
      y = originY;
      dlig = slice.n_rows - origin.y;
      dcol = slice.n_cols - origin.x;
      if(dlig > dcol){
        dmin = dcol;
      }else{
        dmin = dlig;
      }
      #pragma omp for
      for (i=0 ; i<dlig ; i++)
        {
          slice(y,x) += 1;
          x += orientationInv;
          y += 1.;
        }
      for ( x = originX-orientationInv , y=originY-1; x>0. && y>0. ; x -= orientationInv, y -= 1. )
        {
          slice(y,x) += 1;
        }
    }
	else if ( orientation > 0. )
    {
      for ( x = originX, y=originY ; x<widthMinusOne && y<heightMinusOne ; x += 1., y += orientation )
        {
          slice(y,x) += 1;
        }
      for ( x = originX-1., y=originY-orientation ; x>0. && y>0. ; x -= 1., y -= orientation )
        {
          slice(y,x) += 1;
        }
    }
	else if ( orientation > -1. )
    {
      for ( x = originX, y=originY ; x<widthMinusOne && y>0. ; x += 1., y += orientation )
        {
          slice(y,x) += 1;
        }
      for ( x = originX-1., y=originY-orientation ; x>0. && y<heightMinusOne ; x -= 1., y -= orientation )
        {
          slice(y,x) += 1;
        }
    }
	else
    {
      for ( x = originX , y=originY; x>0. && y<heightMinusOne ; x += orientationInv, y += 1. )
        {
          slice(y,x) += 1;
        }
      for ( x = originX-orientationInv , y=originY-1.; x<widthMinusOne && y>0. ; x -= orientationInv, y -= 1. )
        {
          slice(y,x) += 1;
        }
    }
}
