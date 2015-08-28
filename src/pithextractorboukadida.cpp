#include "inc/pithextractorboukadida.h"

#include "inc/billon.h"
#include "inc/coordinate.h"
#include "inc/globalfunctions.h"
#include "inc/globaltimer.h"

#include "inc/mainwindow.h"
#include "ui_mainwindow.h"

#include <iomanip>

static double tallocs = 0;

PithExtractorBoukadida::PithExtractorBoukadida( const int &subWindowWidth, const int &subWindowHeight, const qreal &pithShift, const uint &smoothingRadius,
												const qreal &minWoodPercentage, const Interval<int> &intensityInterval,
												const bool &ascendingOrder, const TKD::ExtrapolationType &extrapolationType,
												const qreal &firstValidSliceToExtrapolate, const qreal &lastValidSliceToExtrapolate ) :
	_subWindowWidth(subWindowWidth), _subWindowHeight(subWindowHeight), _pithShift(pithShift), _smoothingRadius(smoothingRadius),
	_minWoodPercentage(minWoodPercentage), _intensityInterval(intensityInterval), _ascendingOrder(ascendingOrder),
	_extrapolation(extrapolationType), _validSlices(0,0),
	_firstValidSliceToExtrapolate(firstValidSliceToExtrapolate), _lastValidSliceToExtrapolate(lastValidSliceToExtrapolate)
{
}

PithExtractorBoukadida::~PithExtractorBoukadida()
{
}

int PithExtractorBoukadida::subWindowWidth() const
{
	return _subWindowWidth;
}

int PithExtractorBoukadida::subWindowHeight() const
{
	return _subWindowHeight;
}

qreal PithExtractorBoukadida::pithShift() const
{
	return _pithShift;
}

uint PithExtractorBoukadida::smoothingRadius() const
{
	return _smoothingRadius;
}

qreal PithExtractorBoukadida::minWoodPercentage() const
{
	return _minWoodPercentage;
}

Interval<int> PithExtractorBoukadida::intensityInterval() const
{
	return _intensityInterval;
}

bool PithExtractorBoukadida::ascendingOrder() const
{
	return _ascendingOrder;
}

TKD::ExtrapolationType PithExtractorBoukadida::extrapolation() const
{
	return _extrapolation;
}

const Interval<uint> & PithExtractorBoukadida::validSlices() const
{
	return _validSlices;
}

const uint &PithExtractorBoukadida::firstValidSlicesToExtrapolate() const
{
	return _firstValidSliceToExtrapolate;
}

const uint &PithExtractorBoukadida::lastValidSlicesToExtrapolate() const
{
	return _lastValidSliceToExtrapolate;
}

void PithExtractorBoukadida::setSubWindowWidth( const int &width )
{
	_subWindowWidth = width;
}

void PithExtractorBoukadida::setSubWindowHeight( const int &height )
{
	_subWindowHeight = height;
}

void PithExtractorBoukadida::setPithShift( const qreal & shift )
{
	_pithShift = shift;
}

void PithExtractorBoukadida::setSmoothingRadius( const uint &radius )
{
	_smoothingRadius = radius;
}

void PithExtractorBoukadida::setMinWoodPercentage( const qreal &percentage )
{
	_minWoodPercentage = percentage;
}

void PithExtractorBoukadida::setIntensityInterval( const Interval<int> &interval )
{
	_intensityInterval = interval;
}

void PithExtractorBoukadida::setAscendingOrder( const bool &order )
{
	_ascendingOrder = order;
}

void PithExtractorBoukadida::setExtrapolation( const TKD::ExtrapolationType &extrapolationType )
{
	_extrapolation = extrapolationType;
}

void PithExtractorBoukadida::setFirstValidSlicesToExtrapolate( const uint &percentOfSlices )
{
	_firstValidSliceToExtrapolate = percentOfSlices;
}

void PithExtractorBoukadida::setLastValidSlicesToExtrapolate( const uint &percentOfSlices )
{
	_lastValidSliceToExtrapolate = percentOfSlices;
}

void PithExtractorBoukadida::process( Billon &billon, const bool &adaptativeWidth )
{
	const int widthMinusOne = billon.n_cols-1;
	const int heightMinusOne = billon.n_rows-1;
	const int &depth = billon.n_slices;

	const rCoord2D voxelDims(billon.voxelWidth(),billon.voxelHeight());
	const qreal &xDim = voxelDims.x;
	const qreal &yDim = voxelDims.y;

  // Plantage semble venir de valeur de semiSubWindowWidth
  //  printf("===> %d %f\n", _subWindowWidth, xDim);

  // Affichage du nombre de threads
  printf("Nombre de threads pour le calcul de la moelle : %d\n", nbT);

	const int semiSubWindowWidth = qFloor(_subWindowWidth/(2.*xDim));   // Utilisé pour les noeuds : constante utilisateur
	const int semiSubWindowHeight = qFloor(_subWindowHeight/(2.*yDim)); // Utilisé pour les noeuds : constante utilisateur

	Pith &pith = billon._pith; // Récupération de la moelle : structure qui contient les coordonnées (vecteur pour l'ensemble du billon : 1 pt par coupe)

	int k;

	pith.clear();
	pith.resize(depth);

	QVector<qreal> nbLineByMaxRatio( depth, 0. );      // Nb de lignes qui traversent le point central détecté sur chaque coupe divisé par le nb de pixels votants (mesure de qualité)
	QVector<qreal> backgroundProportions( depth, 1. ); // Ratio du nb de pixels > seuil sur nb total de pixels => permet d'éliminer les coupes partielles
  QVector< QVector<iCoord2D> > woodSubWindow( depth, QVector<iCoord2D>(2) ); // Couple de points décrivant le rectangle englobant du tronc dans chaque coupes
  QVector<uint> completeSlices(nbT);                 // Liste des coupes traitées complètement (frontières entre sous-parties du tronc pour traitements multi-threadé)
                                                     // Le dernier élément contient la borne sup des indices des coupes (pour éviter un test)
                                                     // Seule la moitié des valeurs est utilisée pour les coupes frontières

//	qDebug() << "Step 1] Copie du billon";

/* <Time computing> */
	GlobalTimer::getInstance()->start("a) Copie du tronc");
	Billon billonFillBackground(billon); // Déclaration pour fixer les voxels sous le seuil à -1000 -> fond homogène
	GlobalTimer::getInstance()->end();
/* </Time computing> */

//	qDebug() << "Step 2] Fill billon background + proportions";
/* <Time computing> */
	GlobalTimer::getInstance()->start("b) Fill billon background + proportions");
	fillBillonBackground( billonFillBackground, backgroundProportions, woodSubWindow, _intensityInterval, adaptativeWidth ); // Remplit le fond et le vecteur des proportions de voxels du bois
                                                                                                            // adaptativeWidth permet de restreindre le traitement sur les parties non vides de chaque coupe
	GlobalTimer::getInstance()->end();
/* </Time computing> */

//	qDebug() << "Step 3] Detect interval of valid slices";
/* <Time computing> */
	GlobalTimer::getInstance()->start("c) Detect interval of valid slices");
	detectValidSliceInterval( backgroundProportions );    // Génère l'intervalle des coupes conservées
	GlobalTimer::getInstance()->end();
/* </Time computing> */

	const int &firstValidSliceIndex = _validSlices.min();
	const int &lastValidSliceIndex = _validSlices.max();
	const int firstSliceOrdered = _ascendingOrder?firstValidSliceIndex:lastValidSliceIndex;
  // qDebug() << "[ " << firstValidSliceIndex << ", " << lastValidSliceIndex << " ]";

	if ( _validSlices.size() < _smoothingRadius )
	{
		qDebug() << "   => No valid slices, detection stopped";
		return;
	}

	// Calcul de la moelle sur la première coupe valide : traitement différent car pas d'info précédente pour utiliser une sous-matrice
  //	qDebug() << "Step 4] Hough transform on first valid slice";
  /* <Time computing> */
  GlobalTimer::getInstance()->start("d-e) Hough transforms on all slices");
  // double duree, deb = omp_get_wtime();

  // Version parallèle multi-processus du calcul des transformées de Hough
#ifdef PARALLEL
  omp_set_nested(1);
#endif
  #pragma omp parallel if(nbT > 1) num_threads(nbT) private(k)
  {
    int kIncrement = (_ascendingOrder?1:-1);                          // Utilisé pour les noeuds : sens de parcours de coupes le long du noeud ou du billon
                                                                      // Influence sur résultat car utilisation de la coupe précédente pour initier le traitement de la coupe courante
    int nbValidSlices = (lastValidSliceIndex - firstValidSliceIndex + 1) - completeSlices.size() / 2;
    int nbLocalSlices = nbValidSlices / nbT;
    int resteLocalSlices = nbValidSlices % nbT;
#ifdef PARALLEL
    int num = omp_get_thread_num();
#else
    int num = 0;
#endif
    int firstLocalSlice = firstValidSliceIndex + num * nbLocalSlices;
    if(num < resteLocalSlices){
      nbLocalSlices++;
      firstLocalSlice += num;
    }else{
      firstLocalSlice += resteLocalSlices;
    }
    if(num > 0){
      firstLocalSlice += (num + 1) / 2;
    }
    int lastLocalSlice = firstLocalSlice + nbLocalSlices - 1;
    int startLocalSlice; // = kIncrement?firstLocalSlice:lastLocalSlice;
    if(num % 2 == 0){
      if(nbT == 1){
        std::cout << "Il n'y a qu'un thread défini localement" << std::endl;
        kIncrement = (_ascendingOrder?1:-1);
        completeSlices[num] = firstSliceOrdered;
        startLocalSlice = firstSliceOrdered + kIncrement;
      }else{
        kIncrement = -1;
        completeSlices[num] = lastLocalSlice + 1;
        startLocalSlice = lastLocalSlice;
      }
    }else{
      kIncrement = 1;
      completeSlices[num] = firstLocalSlice - 1;
      startLocalSlice = firstLocalSlice;
    }

    // #pragma omp critical
    // {
    //   std::cout << num << " : " << "first : " << firstLocalSlice << " , last : " << lastLocalSlice << " frontier : " << ((num%2==0)?completeSlices[num]:completeSlices[num-1]) << " départ : " << startLocalSlice << " sens : " << kIncrement << std::endl;
    // }

    //  GlobalTimer::getInstance()->start("d) Hough transform on first valid slice");
    if(num % 2 == 0){
      //      pith[startLocalSlice] = transHough( billonFillBackground.slice(startLocalSlice), nbLineByMaxRatio[startLocalSlice],
      //                                          voxelDims, adaptativeWidth?startLocalSlice/static_cast<qreal>(depth):1.0, 2 );

      // std::cout << num << " : " << "calcul de la coupe " << completeSlices[num] << std::endl;
      
      pith[completeSlices[num]] = transHough( billonFillBackground.slice(completeSlices[num]), nbLineByMaxRatio[completeSlices[num]],
                                              voxelDims, adaptativeWidth?completeSlices[num]/static_cast<qreal>(depth):1.0, (nbT>1)?2:1 );
    }

    //  GlobalTimer::getInstance()->end();

    //    std::cout << num << " : attente traitement coupes complètes" << std::endl;

    #pragma omp barrier

    //    std::cout << num << " : barrière CC passée " << std::endl;

    /* </Time computing> */

    /* Calcul de la moelle sur les coupes suivantes
       PB DE SÉQUENTIALITÉ ENTRE LES COUPES DÛ AUX SOUS-FENÊTRES CENTRÉES SUR POSITION MOELLE DE COUPE PRÉCÉDENTE
       => Pour version // on peut :
       1 - se limiter à la fenêtre englobante du tronc sur chaque coupe pour supprimer les dépendances entre les coupes
           !! risque d'avoir des localisations moins précises qu'en utilisant les infos précédentes !!
       2 - répartir les coupes par blocs sur les threads et faire les calculs indépendemment
           !! risque de démarrer un bloc sur une coupe avec noeuds, pouvant fausser la position de la moelle !!
    */

    //	qDebug() <<"Step 5] Hough transform on next valid slices";
    rCoord2D currentPithCoord;
    iCoord2D subWindowStart, subWindowEnd;

    /* <Time computing> */
    // GlobalTimer::getInstance()->start("e) Hough transform on next valid slices");
    // #pragma omp critical
    // {
    //   std::cout << num << " : " << "first : " << firstLocalSlice << " , last : " << lastLocalSlice << " départ : " << startLocalSlice << " sens : " << kIncrement << std::endl;
    // }

    // Traitement des slices restantes
    for ( k = startLocalSlice ; k<=lastLocalSlice && k>=firstLocalSlice ; k += kIncrement )
      {
        //		qDebug() << k ;
        const Slice &currentSlice = billonFillBackground.slice(k);
        const rCoord2D &previousPith = pith[k-kIncrement];
        
        //        std::cout << num << "\tpith coord slice " << k << std::endl;

        subWindowStart.x = qMax(qFloor(previousPith.x-semiSubWindowWidth),0);
        subWindowEnd.x = qMin(qFloor(previousPith.x+semiSubWindowWidth),widthMinusOne);
        subWindowStart.y = qMax(qFloor(previousPith.y-semiSubWindowHeight),0);
        subWindowEnd.y = qMin(qFloor(previousPith.y+semiSubWindowHeight),heightMinusOne);

        //        printf("fenêtre %d %d %d %d <--> (%f , %d) (%f , %d)\n", subWindowStart.y, subWindowStart.x, subWindowEnd.y, subWindowEnd.x, previousPith.x, semiSubWindowWidth, previousPith.y, semiSubWindowHeight);
        // Transformée de Hough sur zone restreinte selon coupe précédente
        currentPithCoord = transHough( currentSlice.submat( subWindowStart.y, subWindowStart.x, subWindowEnd.y, subWindowEnd.x ), nbLineByMaxRatio[k], voxelDims, 1.0, 1 ) + subWindowStart;

        //if ( currentPithCoord.euclideanDistance(previousPith) > _pithShift )
        // Recalcul global si distance entre moelle sur deux coupes consécutives trop importante
        // 
        if ( qSqrt( qPow((currentPithCoord.x-previousPith.x)*xDim,2) + qPow((currentPithCoord.y-previousPith.y)*yDim,2) ) > _pithShift )
          {
            // qDebug() << "\t...  ";
            currentPithCoord = transHough( currentSlice, nbLineByMaxRatio[k], voxelDims, adaptativeWidth?k/static_cast<qreal>(depth):1.0, 1 );
          }
        
        // if(num == 0){
        //   std::cout << "\tpith coord slice " << k << " : " << currentPithCoord.x << " , " << currentPithCoord.y << std::endl;
        // }

        pith[k] = currentPithCoord;
      }

    // #pragma omp critical
    // {
    //   std::cout << "\t" << num << " : " << "first : " << firstLocalSlice << " , last : " << lastLocalSlice << " départ : " << startLocalSlice << " sens : " << kIncrement << std::endl;
    // }

  }
  // duree = omp_get_wtime() - deb;
	GlobalTimer::getInstance()->end();
  // std::cout << " : Traitement des coupes fini : " << duree << "s" << std::endl;

/* </Time computing> */

	// Interpolation des coupes valides dans lesquelles la moelle n'est pas localisée précisément
//	qDebug() << "Step 6] Interpolation of valid slices";
/* <Time computing> */
	GlobalTimer::getInstance()->start("f) Interpolation of valid slices");
	interpolation( pith, nbLineByMaxRatio, _validSlices );
	GlobalTimer::getInstance()->end();
/* </Time computing> */

	// Lissage de la position de la moelle en moyennant avec un certain nb de coupes avant et après (_smoothingRadius)
//	qDebug() << "Step 7] Smoothing of valid slices";
/* <Time computing> */
	GlobalTimer::getInstance()->start("g) Smoothing of valid slices");
	TKD::meanSmoothing<rCoord2D>( pith.begin()+firstValidSliceIndex, pith.begin()+lastValidSliceIndex, _smoothingRadius, false );
	GlobalTimer::getInstance()->end();
/* </Time computing> */

	// Extrapolation des coupes invalides
  // Déduction de la position de la moelle sur les coupes partielles
  //	qDebug() << "Step 8] Extrapolation of unvalid slices";

/* <Time computing> */
	GlobalTimer::getInstance()->start("h) Extrapolation of unvalid slice");
	const int slopeDistance = 3;

	const int firstValidSliceIndexToExtrapolate = firstValidSliceIndex+(lastValidSliceIndex-firstValidSliceIndex)*_firstValidSliceToExtrapolate/100.;
	const int lastValidSliceIndexToExtrapolate = lastValidSliceIndex-(lastValidSliceIndex-firstValidSliceIndex)*_lastValidSliceToExtrapolate/100.;

	const rCoord2D firstValidCoord = pith[firstValidSliceIndexToExtrapolate];
	const rCoord2D lastValidCoord = pith[lastValidSliceIndexToExtrapolate];

	rCoord2D firstValidCoordSlope = (firstValidCoord - pith[firstValidSliceIndexToExtrapolate+slopeDistance])/static_cast<qreal>(slopeDistance);
	// firstValidCoordSlope.x = ((widthMinusOne/2.)-firstValidCoord.x)/static_cast<qreal>(firstValidSliceIndexToExtrapolate);
	const rCoord2D lastValidCoordSlope = (lastValidCoord - pith[lastValidSliceIndexToExtrapolate-slopeDistance])/static_cast<qreal>(slopeDistance);

	switch (_extrapolation)
	{
    case TKD::LINEAR: // position identique à la coupe de référence (1er valide)
//			qDebug() << "  Linear extrapolation";
			for ( k=firstValidSliceIndexToExtrapolate-1 ; k>=0 ; --k )
			{
				pith[k] = firstValidCoord;
			}
			for ( k=lastValidSliceIndexToExtrapolate+1 ; k<depth ; ++k )
			{
				pith[k] = lastValidCoord;
			}
			break;
    case TKD::SLOPE_DIRECTION: // position suivant droite entre 1er valide et slopeDistance
//			qDebug() <<  "  In slope direction extrapolation";
			for ( k=firstValidSliceIndexToExtrapolate-1 ; k>=0 ; --k )
			{
				pith[k] = pith[k+1] + firstValidCoordSlope;
				pith[k].x = qMin(qMax(pith[k].x,0.),static_cast<qreal>(widthMinusOne));
				pith[k].y = qMin(qMax(pith[k].y,0.),static_cast<qreal>(heightMinusOne));
			}
			for ( k=lastValidSliceIndexToExtrapolate+1 ; k<depth ; ++k )
			{
				pith[k] = pith[k-1] + lastValidCoordSlope;
				pith[k].x = qMin(qMax(pith[k].x,0.),static_cast<qreal>(widthMinusOne));
				pith[k].y = qMin(qMax(pith[k].y,0.),static_cast<qreal>(heightMinusOne));
			}
			break;
		case TKD::NO_EXTRAPOLATION:
		default:
//			qDebug() << "  No extrapolation";
			break;
	}
	GlobalTimer::getInstance()->end();
/* </Time computing> */

  QTextStream streamOut(stdout);
  GlobalTimer::getInstance()->print(streamOut);
}

uiCoord2D PithExtractorBoukadida::transHough( const Slice &slice, qreal &lineOnMaxRatio, const rCoord2D &voxelDims, const qreal &adaptativeWidthCoeff, const uint &nbThreads ) const
{
	const int &width = slice.n_cols;
	const int &height = slice.n_rows;

	const int semiWidth = qFloor(width/2.);
	const int semiAdaptativeWidth = qFloor(semiWidth*adaptativeWidthCoeff);
	const int minX = qMax(semiWidth-semiAdaptativeWidth,0);
	const int maxX = qMin(semiWidth+semiAdaptativeWidth,width-1);

	lineOnMaxRatio = 0.;

	if ( semiAdaptativeWidth<1 )
		return uiCoord2D(semiWidth,qFloor(height/2.));

	// Création de la matrice d'accumulation à partir des filtres de Sobel
	arma::Mat<qreal> orientations( height, maxX-minX+1, arma::fill::zeros );
	// QVector< arma::Mat<int> > accuSliceVec( nbThreads, arma::Mat<int>(height, maxX-minX+1, arma::fill::zeros) );
	QVector< arma::Mat<int> > accuSliceVec( 1, arma::Mat<int>(height, maxX-minX+1, arma::fill::zeros) );
	uint nbContourPoints = accumulation( slice.cols(minX,maxX), orientations, accuSliceVec, voxelDims, nbThreads );
  arma::Mat<int> &accuSlice = accuSliceVec[0];

	// Valeur et coordonnée du maximum de accuSlice
	uiCoord2D pithCoord(width/2,height/2);
	if (nbContourPoints)
	{
		lineOnMaxRatio = accuSlice.max(pithCoord.y,pithCoord.x)/static_cast<qreal>(nbContourPoints);
		pithCoord.x += minX;
	}

	return pithCoord;
}

uint PithExtractorBoukadida::accumulation( const Slice &slice, arma::Mat<qreal> & orientations, QVector< arma::Mat<int> > &accuSliceVec,
                                           const rCoord2D &voxelDims, const uint &nbThreads ) const
{
	const uint widthMinusOne = slice.n_cols-1;
	const uint heightMinusOne = slice.n_rows-1;

	const qreal &xDim = voxelDims.x;
	const qreal &yDim = voxelDims.y;
	const qreal voxelRatio = qPow(xDim/yDim,2);

	arma::Col<qreal> sobelNormVec((widthMinusOne-1)*(heightMinusOne-1));
  // arma::Col<qreal>::iterator sobelNormVecIt = sobelNormVec.begin();
  arma::Col<uint> sobelNormSortIndex;
  uint nbContourPoints = (sobelNormVec.n_elem) * 0.4;
  arma::Mat<int> &accuSlice = accuSliceVec[0];

#ifdef PARALLEL
  int numTExt = omp_get_thread_num();
#else
  int numTExt = 0;
#endif

  // Version parallèle multi-processus de l'accumulation pour une droite donnée
  #pragma omp parallel if(nbThreads > 1 && widthMinusOne * heightMinusOne > 1000) num_threads(nbThreads)
  {
    uint i, j, k;
    qreal sobelX, sobelY, norm;

#ifdef PARALLEL
//     // int nbT = omp_get_num_threads();
    int numL = omp_get_thread_num();
#else
//     // int nbT = 1;
    int numL = 0;
#endif
    int firstLine;
    int nextFirstLine;

    // Calcul des bandes d'image
    int nbLines = (heightMinusOne - 1) / 2;
    int reste = (heightMinusOne - 1) % 2;
    firstLine = numL * nbLines + 1;
    if(numL < reste){
      nbLines++;
      firstLine += numL;
    }else{
      firstLine += reste;
    }
    nextFirstLine = firstLine + nbLines;

    // if(nbThreads > 1){
    //   std::cout << numTExt << " " << numL << " : calculs d'accumulation -> " << firstLine << " , " <<  nextFirstLine << std::endl;
    // }

    // Calcul des filtres de Sobel
    #pragma omp for collapse(2)
    for ( i=1 ; i<widthMinusOne ; ++i )
      {
        for ( j=1 ; j<heightMinusOne ; ++j )
          {
            // Calcul des deux filtres de contours en vertical et horizontal
            sobelX = slice( j-1, i-1 ) - slice( j-1, i+1 ) + 2* (slice( j, i-1 ) - slice( j, i+1 )) + slice( j+1, i-1 ) - slice( j+1, i+1 );
            sobelY = slice( j+1, i-1 ) - slice( j-1, i-1 ) + 2 * (slice( j+1, i ) - slice( j-1, i )) + slice( j+1, i+1 ) - slice( j-1, i+1 );
            // Déduction de l'orientation de la normale et de sa norme à partir des deux filtres (vertical et horizontal)
            orientations(j,i) = qFuzzyIsNull(sobelX) ? 9999999999./1. : sobelY/sobelX*voxelRatio;
            norm = qPow(sobelX,2) + qPow(sobelY,2);
            // *sobelNormVecIt++ = norm;
            sobelNormVec((j-1) * (widthMinusOne-1) + i - 1) = norm;
          }
      }
    
    // Tri des normes et conservation d'un certain pourcentage des plus élevées (40% des normes non nulles)
    #pragma omp single
    {
      sobelNormSortIndex = arma::sort_index( sobelNormVec, "descend" );
      // std::cout << numTExt << " " << numL << " : tri des index fait" << std::endl;
      // nbContourPoints = (sobelNormVec.n_elem) * 0.4;
    }
    
    if(nbThreads > 1) {
      for ( k=0 ; k<nbContourPoints ; ++k ) {
        i = (sobelNormSortIndex[k] % (widthMinusOne-1)) + 1;
        j = (sobelNormSortIndex[k] / (widthMinusOne-1)) + 1;
        drawLinePart( accuSliceVec[0], uiCoord2D(i,j), -orientations(j,i), firstLine, nextFirstLine );
      }
    }else{
      for ( k=0 ; k<nbContourPoints ; ++k )
        {
          i = (sobelNormSortIndex[k] % (widthMinusOne-1)) + 1;
          j = (sobelNormSortIndex[k] / (widthMinusOne-1)) + 1;
          drawLine( accuSliceVec[numL], uiCoord2D(i,j), -orientations(j,i) );
        }
    }
    
    // #pragma omp barrier
    // //    std::cout << numTExt << " " << numL << " : accumulation finale" << std::endl;
    
    // if(nbThreads > 1)
    //   {
    //     #pragma omp for
    //     for(j=0; j<slice.n_rows; ++j)
    //       {
    //         for(i=0; i<slice.n_cols; ++i)
    //           {
    //             for(k=1; k<nbThreads; ++k)
    //               {
    //                 accuSlice(j, i) += accuSliceVec[k](j, i);
    //               }
    //           }
    //       }
    //   }
  }
  //  std::cout << numTExt << " : fin accumulation" << std::endl;

	return nbContourPoints;
}

void PithExtractorBoukadida::drawLine(arma::Mat<int> &slice, const iCoord2D &origin, const qreal &orientation ) const
{
	const int heightMinusOne = slice.n_rows-1;
	const int widthMinusOne = slice.n_cols-1;
	const int originX = origin.x;
	const int originY = origin.y;
	const qreal orientationInv = 1./orientation;

	qreal x, y;

	if ( orientation >= 1. )
	{
		for ( x = originX , y=originY; x<widthMinusOne && y<heightMinusOne ; x += orientationInv, y += 1. )
		{
			slice(y,x) += 1;
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

void avoidMiddleFloat(double *x)
{
  double ent;
  double dec = fabs(modf(*x, &ent));
  if(dec>=.5-(1e-6) && dec<=.5+(1e-6)){
    *x -= copysign(1e-5, *x);
  }
}

void PithExtractorBoukadida::drawLinePart(arma::Mat<int> &slice, const iCoord2D &origin, const qreal &orientation, const int firstLine, const int nextFirstLine ) const
{
	const int heightMinusOne = slice.n_rows-1;
	const int widthMinusOne = slice.n_cols-1;
	const int originX = origin.x;
	const int originY = origin.y;
	const qreal orientationInv = 1./orientation;

	qreal x, y;

  // Placement du point de départ sur bords haut-gauche-droite de sous-image
  x = originX + orientationInv * (firstLine - originY);
  if(x < 0){
    y = firstLine - x * orientation;
    x = 0;
  }else{
    if(x >= widthMinusOne){
      y = firstLine - (x - widthMinusOne + 1) * orientation;
      x = widthMinusOne - 1;
    }else{
      y = firstLine;
    }
  }

  // qDebug() << "  line init  " << x << " , " << y;
  if(y>=firstLine && y<nextFirstLine){
    if ( orientation > 1. )
      {
        avoidMiddleFloat(&y);
        // Parcours jusqu'à sortie de la sous-image
        for ( ; x<widthMinusOne && y<nextFirstLine ; x += orientationInv, y += 1. )
          {
            // if(y<0 || y>=nextFirstLine || x<0 || x>=widthMinusOne){
            //   qDebug() << "Dépassement image cas 1 : " << x << " , " << y << " -> " << firstLine << " , " << nextFirstLine << " -> " << originX << " , " << originY << " | " << orientation << " , " << orientationInv;
            // }
            slice(y,x) += 1;
          }
      }
    else if ( orientation > 0. )
      {
        avoidMiddleFloat(&x);
        // Parcours jusqu'à sortie de la sous-image
        for ( ; x<widthMinusOne && y<nextFirstLine ; x += 1., y += orientation )
          {
            // if(y<0 || y>=nextFirstLine || x<0 || x>=widthMinusOne){
            //   qDebug() << "Dépassement image cas 2 : " << x << " , " << y << " -> " << firstLine << " , " << nextFirstLine << " -> " << originX << " , " << originY << " | " << orientation << " , " << orientationInv;
            // }
            slice(y,x) += 1;
          }
      }
    else if ( orientation < -1. )
      {
        avoidMiddleFloat(&y);
        // Parcours jusqu'à sortie de la sous-image
        for ( ; x>=0 && y<nextFirstLine ; x += orientationInv, y += 1. )
          {
            // if(y<0 || y>=nextFirstLine || x<0 || x>=widthMinusOne){
            //   qDebug() << "Dépassement image cas 3 : " << x << " , " << y << " -> " << firstLine << " , " << nextFirstLine << " -> " << originX << " , " << originY << " | " << orientation << " , " << orientationInv;
            // }
            slice(y,x) += 1;
          }
      }
    else
      {
        avoidMiddleFloat(&x);
        // Parcours jusqu'à sortie de la sous-image
        for ( ; x>=0 && y<nextFirstLine ; x -= 1., y -= orientation )
          {
            // if(y<0 || y>=nextFirstLine || x<0 || x>=widthMinusOne){
            //   qDebug() << "Dépassement image cas 4 : " << x << " , " << y << " -> " << firstLine << " , " << nextFirstLine << " -> " << originX << " , " << originY << " | " << orientation << " , " << orientationInv;
            // }
            slice(y,x) += 1;
          }
      }
  }
}

/*
  Interpolation linéaire de la localisation de la moelle entre coupes dont nbLineByMaxRatio sous un seuil calculé automatiquement via les quartiles
 */
void PithExtractorBoukadida::interpolation( Pith &pith, const QVector<qreal> &nbLineByMaxRatio, const Interval<uint> &sliceIntervalToInterpolate ) const
{
	const uint &firstSlice = sliceIntervalToInterpolate.min();
	const uint &lastSlice = sliceIntervalToInterpolate.max();

	QVector<qreal> nbLineByMaxRatioSorting = nbLineByMaxRatio.mid( firstSlice, lastSlice-firstSlice+1 );
	qSort(nbLineByMaxRatioSorting);

	const qreal &quartile1 = nbLineByMaxRatioSorting[ 0.25 * nbLineByMaxRatioSorting.size() ];
	const qreal &quartile3 = nbLineByMaxRatioSorting[ 0.75 * nbLineByMaxRatioSorting.size() ];
	const qreal interpolationThreshold = quartile1 - 0.5 * ( quartile3 - quartile1 );
  qDebug() << "!!!!! SEUIL INTERPOLATION : " << interpolationThreshold << " (" << quartile1 << "," << quartile3 << ")" << endl;

	uint startSliceIndex, newK, startSliceIndexMinusOne;
	rCoord2D interpolationStep, currentInterpolatePithCoord;
	for ( uint k=firstSlice+1 ; k<lastSlice ; ++k )
	{
		if ( nbLineByMaxRatio[k] < interpolationThreshold )
		{
			startSliceIndex = k++;
			startSliceIndexMinusOne = startSliceIndex?startSliceIndex-1:0;

			while ( k<=lastSlice && nbLineByMaxRatio[k] < interpolationThreshold ) ++k;
			if ( k>startSliceIndex ) --k;

//			qDebug() << "Interpolation [" << startSliceIndex << ", " << k << "]" ;

			interpolationStep = k<lastSlice ? (pith[k+1] - pith[startSliceIndexMinusOne]) / static_cast<qreal>( k+1-startSliceIndexMinusOne )
				: rCoord2D(0,0);

			currentInterpolatePithCoord = interpolationStep + pith[startSliceIndexMinusOne];

			for ( newK = startSliceIndex ; newK <= k ; ++newK, currentInterpolatePithCoord += interpolationStep )
			{
				pith[newK] = currentInterpolatePithCoord;
			}
		}
	}
}

void PithExtractorBoukadida::fillBillonBackground( Billon &billonToFill, QVector<qreal> &backgroundProportions, QVector< QVector<iCoord2D> > &woodSubWindow,
												   const Interval<int> &intensityInterval, const bool &adaptativeWidth ) const
{
	const int &width = billonToFill.n_cols;
	const int &height = billonToFill.n_rows;
	const int &nbSlices = billonToFill.n_slices;
	const int &minIntensity = intensityInterval.min();
	const int &maxIntensity = intensityInterval.max();

	const int semiWidth = qFloor(width/2.);

	Slice::col_iterator begin, end;
	int k, semiAdaptativeWidth;
	int iMin, iMax;
	qreal adaptativeWidthCoeff, currentProp;
	__billon_type__ val;
  int i, j;
  bool inWood;

	iMin = 0;
	iMax = width-1;
	for ( k=0 ; k<nbSlices ; ++k )
	{
    iCoord2D &topLeft = woodSubWindow[k][0];
    iCoord2D &bottomRight = woodSubWindow[k][1];
    
    double tlx = topLeft.x = width;
    double tly = topLeft.y = height;
    double brx = bottomRight.x = 0;
    double bry = bottomRight.y = 0;

		Slice &currentSlice = billonToFill.slice(k);
		if ( adaptativeWidth )
		{
			adaptativeWidthCoeff = k/static_cast<qreal>(nbSlices);
			semiAdaptativeWidth = qRound(semiWidth*adaptativeWidthCoeff);
			iMin = semiWidth-semiAdaptativeWidth;
			iMax = semiWidth+semiAdaptativeWidth-1;
			if ( semiAdaptativeWidth<2 ) continue;
		}
		begin = currentSlice.begin_col(iMin);
		end = currentSlice.end_col(iMax);
		currentProp = 0.;
		// while ( begin != end )
		// {
		// 	val = qMax( qMin( *begin, maxIntensity ), minIntensity );
		// 	currentProp += (val == minIntensity || val == maxIntensity);
		// 	*begin++ = val;
		// }
    #pragma omp parallel for private(j,val,inWood) reduction(min:tlx,tly) reduction(max:brx,bry) reduction(+:currentProp) num_threads(nbT)
    for ( i=iMin; i<=iMax; ++i ){
      for ( j=0; j<height; ++j ){
        val = qMax( qMin( currentSlice(j, i), maxIntensity ), minIntensity );
        inWood = !(val == minIntensity || val == maxIntensity);
        if ( inWood ){ // Mise à jour des min-max en x et y
          if ( topLeft.x > i ){
            // topLeft.x = i;
            tlx = i;
          }
          if ( bottomRight.x < i ){
            // bottomRight.x = i;
            brx = i;
          }
          if ( topLeft.y > j ){
            // topLeft.y = j;
            tly = j;
          }
          if ( bottomRight.y < j ){
            // bottomRight.y = j;
            bry = j;
          }
        }else{
          currentProp++;
        }
        currentSlice(j, i) = val;
      }
    }
    topLeft.x = tlx;
    topLeft.y = tly;
    bottomRight.x = brx;
    bottomRight.y = bry;

    // Centrage de la sous-fenêtre
    if ( topLeft.x < width - bottomRight.x - 1 ){
      topLeft.x = width - bottomRight.x - 1;
    }else{
      bottomRight.x = width - topLeft.x;
    }
    if ( topLeft.y < height - bottomRight.y - 1 ){
      topLeft.y = height - bottomRight.y - 1;
    }else{
      bottomRight.y = height - topLeft.y;
    }
		backgroundProportions[k] = currentProp / static_cast<qreal>(height*(iMax-iMin+1));
	}
}

void PithExtractorBoukadida::detectValidSliceInterval( const QVector<qreal> &backgroundProportions )
{
	const uint &nbSlices = backgroundProportions.size();
	const qreal backgroundPercentage = (100.-_minWoodPercentage)/100.;
	uint sliceIndex;;

	sliceIndex = 0;
	while ( sliceIndex<nbSlices && backgroundProportions[sliceIndex] > backgroundPercentage ) sliceIndex++;
	_validSlices.setMin(qMin(sliceIndex,nbSlices-1));

	const uint &minValid = _validSlices.min();
	sliceIndex = nbSlices-1;
	while ( sliceIndex>minValid && backgroundProportions[sliceIndex] > backgroundPercentage ) sliceIndex--;
	_validSlices.setMax(sliceIndex);
}
