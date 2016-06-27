// FD_Data.H
#if !defined  FD_DATA_H
#define FD_DATA_H

#define MAX_HAAR_FEATURE	3

typedef struct FdRect	
{
	int	x;
	int	y;
	int	width;
	int	height;
	float view; // face view class
	int neighbors;
	double confidence;
} FdRect;

typedef struct ROI_Rect	
{
	int	x;
	int	y;
	int	width;
	int	height;
} ROI_Rect;

typedef struct EdRect	
{
	int	lx;
	int	ly;
	int	rx;
	int	ry;
} EdRect;

__inline  FdRect  fdRect( int x, int y, int width, int height );
__inline  FdRect  fdRect( int x, int y, int width, int height )
{
    FdRect r;
    r.x = x;
    r.y = y;
    r.width = width;
    r.height = height;
    return r;
}

typedef struct FdSize
{
    int width;
    int height;
}
FdSize;

__inline  FdSize  fdSize( int width, int height );
__inline  FdSize  fdSize( int width, int height )
{
    FdSize s;
    s.width = width;
    s.height = height;
    return s;
}

typedef struct HaarFeature
{
    int  tilted;
    struct
    {
        FdRect r;
        float weight;
    } rect[MAX_HAAR_FEATURE];
} HaarFeature;

typedef struct HidFeature	
{
	struct
	{
		int	*p0, *p1, *p2, *p3;
		float weight;
	} rect[MAX_HAAR_FEATURE];
} HidFeature;

typedef struct HidClassifier	
{
	int	count;
	HaarFeature	*origFeature;
	HidFeature	*feature;
	float	*threshold;
	int	*left;
	int	*right;
	float	*alpha;
} HidClassifier;

typedef struct HidStage	
{
	int	count;
	float	threshold;
	HidClassifier	*classifier;
	int	two_rects;
} HidStage;

typedef struct HidCascade	
{
    int  headerSize;
    int  count;
    int  is_stump_based;
    int  has_tilted_features;
    FdSize origWindowSize;
    FdSize realWindowSize;
    double scale, invWindowArea;
    HidStage* stageClassifier;
    int *sum;
    int *tiltsum;
    double *sqsum;
    double *pq0, *pq1, *pq2, *pq3;
    int *p0, *p1, *p2, *p3;
} HidCascade;

typedef struct HaarClassifier
{
    int count;
    HaarFeature* haarFeature;
    float* threshold;
    int* left;
    int* right;
    float* alpha;
}
HaarClassifier;

typedef struct HaarStage
{
    int  count;
    float threshold;
    HaarClassifier* classifier;
}
HaarStage;

typedef struct HaarCascade
{
    int  count;
    FdSize origWindowSize;
    HaarStage* stageClassifier;
}
HaarCascade;

typedef struct FdAvgComp
{
	FdRect	rect;
	float	neighbors;
	int		eyeLocation; // 0 = left, 1 = right
	float	view;
	double  confidence;
} FdAvgComp;

/*
class CComp : public CObject
{
	public:
		FdRect	rect;
		int		neighbors;
		int		eyeLocation; // 0 = left, 1 = right
		int		view;
		double  confidence;
	public:
		CComp();
		virtual ~CComp();
};
*/
#endif 
