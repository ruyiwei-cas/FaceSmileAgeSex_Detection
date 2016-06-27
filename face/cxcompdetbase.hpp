#ifndef _CX_COMPDETBASE_HPP_
#define _CX_COMPDETBASE_HPP_

#include <opencv/cxcore.h>

class CxCompDetBase
{
public:
    CxCompDetBase() {}

    virtual ~CxCompDetBase() {}

  	virtual bool detect( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, float angle=0 ) = 0;

	virtual bool track( const IplImage* image, CvRect* rect, CvPoint2D32f points[], float parameters[]=NULL, float angle=0 ) = 0;

    virtual CvPoint2D32f getPoint( int comp ) const = 0;
};

#endif
