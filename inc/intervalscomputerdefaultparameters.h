#ifndef INTERVALSCOMPUTERDEFAULTPARAMETERS_H
#define INTERVALSCOMPUTERDEFAULTPARAMETERS_H

#define DEFAULT_MASK_RADIUS 2
#define DEFAULT_MINIMUM_WIDTH_OF_NEIGHBORHOOD 10
#define DEFAULT_MINIMUM_WIDTH_OF_INTERVALS 10
#define DEFAULT_PERCENTAGE_FOR_MAXIMUM_CANDIDATE 0.01

// Types de coupe possibles
namespace SmoothingType {
	enum SmoothingType {
		_SMOOTHING_TYPE_MIN_ = -1,
		NONE,
		MEANS,
		GAUSSIAN,
		_SMOOTHING_TYPE_MAX_
	};
}

#endif // INTERVALSCOMPUTERDEFAULTPARAMETERS_H