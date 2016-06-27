#include "Common_Func.h"

bool bDebug = true;
char *sLogFileName = "FaceDetection_Log.txt";

bool CLEAR_DEBUGLOG()
{
	if(!bDebug)  return false;

	FILE *fLogFile;
	fLogFile = fopen(sLogFileName, "w");
	fprintf(fLogFile, "-----------------  Pose Estimater begins  -------------------\n");
	fclose(fLogFile);
	return true;
}
bool WRITE_DEBUGLOG(char*  sInfo)
{
	if(!bDebug)  return false;

	FILE *fLogFile;
	fLogFile = fopen(sLogFileName, "a+");
    fseek(fLogFile, 0, SEEK_END);
 
	fprintf(fLogFile, sInfo);
	fclose(fLogFile);
	return true;
}

bool WRITE_DEBUGLOG(char*  sInfo, int data)
{
	if(!bDebug)  return false;

	char sText[512];
	memset(sText, 0, 512);
	sprintf(sText, "%s %d\n", sInfo,data);
	
	bool bResult = WRITE_DEBUGLOG(sText);

	return bResult;
}
bool WRITE_DEBUGLOG(char*  sInfo, float data)
{
	if(!bDebug)  return false;

	char sText[512];
	memset(sText, 0, 512);
	sprintf(sText, "%s %f\n", sInfo,data);
	
	bool bResult = WRITE_DEBUGLOG(sText);

	return bResult;
}

bool WRITE_DEBUGLOG(char*  sInfo, double data)
{
	if(!bDebug)  return false;

	char sText[512];
	memset(sText, 0, 512);
	sprintf(sText, "%s %f\n", sInfo,data);
	
	bool bResult = WRITE_DEBUGLOG(sText);

	return bResult;
}

void HV_SUM_OFFSETS_Func( int *p0, int *p1, int *p2, int *p3, CvRect rect, int step)
{
    /* (x, y) */                                                          
    (*p0) = (rect).x + (step) * (rect).y;
    /* (x + w, y) */                                                      
    (*p1) = (rect).x + (rect).width + (step) * (rect).y;
    /* (x, y + h) */                                                      
    (*p2) = (rect).x + (step) * ((rect).y + (rect).height);
    /* (x + w, y + h) */                                                  
    (*p3) = (rect).x + (rect).width + (step) * ((rect).y + (rect).height);
}


HvStumpClassifier* CreateStumpClassifier()
{
	HvStumpClassifier * Stumpclassifier = (HvStumpClassifier*) cvAlloc( sizeof( HvStumpClassifier) );
    memset( (void*) Stumpclassifier, 0, sizeof( HvStumpClassifier ) );
	
	return Stumpclassifier;
}
void ReleaseStumpClassifier( HvStumpClassifier** Stumpclassifier )
{
    cvFree( (void**) Stumpclassifier );
    *Stumpclassifier = 0;
}	

HvRealStumpClassifier* CreateRealStumpClassifier()
{
	HvRealStumpClassifier * Realstumpclassifier = (HvRealStumpClassifier*) cvAlloc( sizeof( HvRealStumpClassifier) );
    memset( (void*) Realstumpclassifier, 0, sizeof( HvRealStumpClassifier ) );
	
	return Realstumpclassifier;
}
void ReleaseRealStumpClassifier( HvRealStumpClassifier** Realstumpclassifier )
{
    cvFree( (void**) Realstumpclassifier );
    *Realstumpclassifier = 0;
}	

HvFeature hvHaarFeature( const char* desc,
						int x0, int y0, int w0, int h0, float wt0,
						int x1, int y1, int w1, int h1, float wt1,
						int x2, int y2, int w2, int h2, float wt2)
{
    HvFeature hf;

    assert( HV_FEATURE_MAX >= 3 );
    assert( strlen( desc ) < HV_FEATURE_DESC_MAX );

    strcpy( &(hf.desc[0]), desc );
    hf.tilted = ( hf.desc[0] == 't' );
    
    hf.rect[0].r.x = x0;
    hf.rect[0].r.y = y0;
    hf.rect[0].r.width  = w0;
    hf.rect[0].r.height = h0;
    hf.rect[0].weight   = wt0;
    
    hf.rect[1].r.x = x1;
    hf.rect[1].r.y = y1;
    hf.rect[1].r.width  = w1;
    hf.rect[1].r.height = h1;
    hf.rect[1].weight   = wt1;
    
    hf.rect[2].r.x = x2;
    hf.rect[2].r.y = y2;
    hf.rect[2].r.width  = w2;
    hf.rect[2].r.height = h2;
    hf.rect[2].weight   = wt2;
    
    return hf;
}

void Sort_32f( float* array, int length, int aux )
// float*를 작은 것부터 sorting.
{                                                                       
    const int bubble_level = 8;                                         
                                                                        
    struct                                                              
    {                                                                   
        int lb, ub;                                                     
    }                                                                   
    stack[48];                                                          
                                                                        
    int sp = 0;                                                         
                                                                        
    float   temp;                                                           
    float   lb_val;                                                         
                                                                        
//	aux = aux;                                                          
                                                                        
    stack[0].lb = 0;                                                    
    stack[0].ub = length - 1;                                           
                                                                        
    while( sp >= 0 )                                                    
    {                                                                   
        int lb = stack[sp].lb;                                          
        int ub = stack[sp--].ub;                                        
                                                                        
        for(;;)                                                         
        {                                                               
            int diff = ub - lb;                                         
            if( diff < bubble_level )	// 개수가 8개 미만이면, qsort안 쓰고, 단순 비교하여 교환한다.                                   
            {                                                           
                int i, j;                                               
                float* arr = array + lb;                                    
                                                                        
                for( i = diff; i > 0; i-- )                             
                {                                                       
                    int f = 0;                                          
                    for( j = 0; j < i; j++ )                            
                        if( less_than( arr[j+1], arr[j] ))              
                        {                                               
                            temp = arr[j];                              
                            arr[j] = arr[j+1];                          
                            arr[j+1] = temp;                            
                            f = 1;                                      
                        }                                               
                    if( !f ) break;                                     
                }                                                       
                break;                                                  
            }                                                           
            else						// 개수가 8개 이상이면, qsort를 쓴다.                                               
            {                                                           
                // select pivot and exchange with 1st element       
                int  m = lb + (diff >> 1);                              
                int  i = lb + 1, j = ub;                                
                                                                        
                lb_val = array[m];                                      
                                                                        
                array[m]  = array[lb];                                  
                array[lb] = lb_val;                                     
                                                                        
                // partition into two segments                        
                for(;;)                                                 
                {                                                       
                    for( ;i < j && less_than(array[i], lb_val); i++ );  
                    for( ;j >= i && less_than(lb_val, array[j]); j-- ); 
                                                                        
                    if( i >= j ) break;                                 
                    temp = array[i];                                    
                    array[i++] = array[j];                              
                    array[j--] = temp;                                  
                }                                                       
                                                                        
                // pivot belongs in A[j]                              
                array[lb] = array[j];                                   
                array[j]  = lb_val;                                     
                                                                        
                // keep processing smallest segment, and stack largest
                if( j - lb <= ub - j )                                  
                {                                                       
                    if( j + 1 < ub )                                    
                    {                                                   
                        stack[++sp].lb   = j + 1;                       
                        stack[sp].ub = ub;                              
                    }                                                   
                    ub = j - 1;                                         
                }                                                       
                else                                                    
                {                                                       
                    if( j - 1 > lb)                                     
                    {                                                   
                        stack[++sp].lb = lb;                            
                        stack[sp].ub = j - 1;                           
                    }                                                   
                    lb = j + 1;                                         
                }                                                       
            }                                                           
        }                                                               
    }                                                                   
}

void SortIndexedValArray_16s( short* array, int length, HvValArray* aux )
// SortIndexedValArray_16s( idx의 한 row (한 개 feature에 대한), image의 수: 200, HvValArray* aux );
//				 aux->data: val의 한 row (한 개 feature에 대한 feature의 weighted sum), aux->step: 4
// 한 개 feature에 대한 전체 training image의 weighted sum을 작은 것부터 sorting한 것.
{                                                                       
    const int bubble_level = 8;                                         
                                                                        
    struct                                                              
    {                                                                   
        int lb, ub;                                                     
    }                                                                   
    stack[48];                                                          
                                                                        
    int sp = 0;                                                         
                                                                        
    short temp;                                                           
    short lb_val;                                                         
                                                                        
    stack[0].lb = 0;                                                    
    stack[0].ub = length - 1;                                           
                                                                        
    while( sp >= 0 )                                                    
    {                                                                   
        int lb = stack[sp].lb;                                          
        int ub = stack[sp--].ub;                                        
                                                                        
        for(;;)                                                         
        {                                                               
            int diff = ub - lb;                                         
            if( diff < bubble_level )     // 개수가 8개 미만이면, qsort안 쓰고, 단순 비교하여 교환한다.                              
            {                                                           
                int i, j;                                               
                short* arr = array + lb;                                    
                                                                        
                for( i = diff; i > 0; i-- )                             
                {                                                       
                    int f = 0;                                          
                    for( j = 0; j < i; j++ )                            
                        if( CMP_VALUES( arr[j+1], arr[j] )) // "arr[j+1]의 내용" < "arr[j]의 내용" 이면,              
                        {									// 둘을 교환.                                          
                            temp = arr[j];                              
                            arr[j] = arr[j+1];                          
                            arr[j+1] = temp;                            
                            f = 1;                                      
                        }                                               
                    if( !f ) break;                                     
                }                                                       
                break;                                                  
            }                                                           
            else							// 개수가 8개 이상이면, qsort를 쓴다.                                                    
            {                                                           
                // select pivot and exchange with 1st element         
                int  m = lb + (diff >> 1);	// pivot                           
                int  i = lb + 1, j = ub;	// i는 두번째 것. j는 마지막 것.                          
                                                                        
                lb_val = array[m];			// lb(첫번째 것)의 array값과 m의 array값을 교환.                                
                array[m]  = array[lb];                                  
                array[lb] = lb_val;        
				
                // partition into two segments: i=j                   
                for(;;)                                                 
                {                                                       
                    for( ;i < j && CMP_VALUES(array[i], lb_val); i++ );  // i 찾기.
                    for( ;j >= i && CMP_VALUES(lb_val, array[j]); j-- ); // j 찾기.
  
                    if( i >= j ) break;                                 
                    temp = array[i];                                    
                    array[i++] = array[j];                              
                    array[j--] = temp;        
                }        

             
                // pivot belongs in A[j]     // lb의 array값과 j의 array값을 교환.                        
                array[lb] = array[j];                                   
                array[j]  = lb_val;                                     
                                                                        
                // keep processing smallest segment, and stack largest
                if( j - lb <= ub - j )                                  
                {                                                       
                    if( j + 1 < ub )                                    
                    {                                                   
                        stack[++sp].lb = j + 1;                       
                        stack[sp].ub = ub;                              
                    }                                                   
                    ub = j - 1;                                         
                }                                                       
                else                                                    
                {                                                       
                    if( j - 1 > lb)                                     
                    {                                                   
                        stack[++sp].lb = lb;                            
                        stack[sp].ub = j - 1;                           
                    }                                                   
                    lb = j + 1;                                         
                }                                                       
            }                                                           
        }                                                               
    }                                                                   
}
void SortIndexedValArray_32s( int* array, int length, HvValArray* aux )
// SortIndexedValArray_16s( idx의 한 row (한 개 feature에 대한), image의 수: 200, HvValArray* aux );
//				 aux->data: val의 한 row (한 개 feature에 대한 feature의 weighted sum), aux->step: 4
// 한 개 feature에 대한 전체 training image의 weighted sum을 작은 것부터 sorting한 것.
{                                                                       
    const int bubble_level = 8;                                         
                                                                        
    struct                                                              
    {                                                                   
        int lb, ub;                                                     
    }                                                                   
    stack[48];                                                          
                                                                        
    int sp = 0;                                                         
                                                                        
    int temp;                                                           
    int lb_val;                                                         
                                                                        
    stack[0].lb = 0;                                                    
    stack[0].ub = length - 1;                                           
                                                                        
    while( sp >= 0 )                                                    
    {                                                                   
        int lb = stack[sp].lb;                                          
        int ub = stack[sp--].ub;                                        
                                                                        
        for(;;)                                                         
        {                                                               
            int diff = ub - lb;                                         
            if( diff < bubble_level )     // 개수가 8개 미만이면, qsort안 쓰고, 단순 비교하여 교환한다.                              
            {                                                           
                int i, j;                                               
                int* arr = array + lb;                                    
                                                                        
                for( i = diff; i > 0; i-- )                             
                {                                                       
                    int f = 0;                                          
                    for( j = 0; j < i; j++ )                            
                        if( CMP_VALUES( arr[j+1], arr[j] )) // "arr[j+1]의 내용" < "arr[j]의 내용" 이면,              
                        {									// 둘을 교환.                                          
                            temp = arr[j];                              
                            arr[j] = arr[j+1];                          
                            arr[j+1] = temp;                            
                            f = 1;                                      
                        }                                               
                    if( !f ) break;                                     
                }                                                       
                break;                                                  
            }                                                           
            else							// 개수가 8개 이상이면, qsort를 쓴다.                                                    
            {                                                           
                // select pivot and exchange with 1st element         
                int  m = lb + (diff >> 1);	// pivot                           
                int  i = lb + 1, j = ub;	// i는 두번째 것. j는 마지막 것.                          
                                                                        
                lb_val = array[m];			// lb(첫번째 것)의 array값과 m의 array값을 교환.                                
                array[m]  = array[lb];                                  
                array[lb] = lb_val;        
				
                // partition into two segments: i=j                   
                for(;;)                                                 
                {                                                       
                    for( ;i < j && CMP_VALUES(array[i], lb_val); i++ );  // i 찾기.
                    for( ;j >= i && CMP_VALUES(lb_val, array[j]); j-- ); // j 찾기.
  
                    if( i >= j ) break;                                 
                    temp = array[i];                                    
                    array[i++] = array[j];                              
                    array[j--] = temp;        
                }        

             
                // pivot belongs in A[j]     // lb의 array값과 j의 array값을 교환.                        
                array[lb] = array[j];                                   
                array[j]  = lb_val;                                     
                                                                        
                // keep processing smallest segment, and stack largest
                if( j - lb <= ub - j )                                  
                {                                                       
                    if( j + 1 < ub )                                    
                    {                                                   
                        stack[++sp].lb = j + 1;                       
                        stack[sp].ub = ub;                              
                    }                                                   
                    ub = j - 1;                                         
                }                                                       
                else                                                    
                {                                                       
                    if( j - 1 > lb)                                     
                    {                                                   
                        stack[++sp].lb = lb;                            
                        stack[sp].ub = j - 1;                           
                    }                                                   
                    lb = j + 1;                                         
                }                                                       
            }                                                           
        }                                                               
    }                                                                   
}

float EvalFastFeature( HvFastFeature* feature, int* sum, int* tilted )
{
    int* img = NULL;
    int i = 0;
    float ret = 0.0F;
    
    assert( feature );
    
    img = ( feature->tilted ) ? tilted : sum;

    assert( img );

    for( i = 0; feature->rect[i].weight != 0.0F && i < HV_FEATURE_MAX; i++ )
    {
        ret += feature->rect[i].weight *
            ( img[feature->rect[i].p0] - img[feature->rect[i].p1] -
              img[feature->rect[i].p2] + img[feature->rect[i].p3] );
    }

    return ret;
}

void GetSortedIndices(CvMat* val,  CvMat* idx, int sortcols )
{
    int idxtype = 0;
    uchar* data = NULL;
    size_t istep = 0;
    size_t jstep = 0;
    int i = 0, j = 0;

    HvValArray va;

    assert( idx != NULL );
    assert( val != NULL );

    idxtype = CV_MAT_TYPE( idx->type );
    assert( idxtype == CV_16SC1 || idxtype == CV_32SC1 || idxtype == CV_32FC1 ); // CV_16SC1
    assert( CV_MAT_TYPE( val->type ) == CV_32FC1 ); // CV_32FC1

    if( sortcols )
    {
        assert( idx->rows == val->cols );
        assert( idx->cols == val->rows );
        istep = CV_ELEM_SIZE( val->type );
        jstep = val->step;
    }
    else
    {
        assert( idx->rows == val->rows );	// feature 수: 52,297
        assert( idx->cols == val->cols );	// training image 수: 200
        istep = val->step;					// 800 = 4(byte)*200(samples:cols)
        jstep = CV_ELEM_SIZE( val->type );	// 4 (float:4 bytes)
    }

    va.data = val->data.ptr; // val의 pointer를 받음.
    va.step = jstep;		 // 4

    for( i = 0; i < idx->rows; i++ )	// feature 수
    {
        for( j = 0; j < idx->cols; j++ )// training image 수
        {
            CV_MAT_ELEM( *idx, int, i, j ) = (int) j; // 모든 column에 0,1,자연수 넣음.
        }

		SortIndexedValArray_32s( (int*) (idx->data.ptr + i * idx->step), idx->cols, &va );

        va.data += istep; // 800
    }
}

void GetSortedIndices(CvMat* val,  CvMat* idx, int sortrows, int sortcols)
{
    int idxtype = 0;
    uchar* data = NULL;
    size_t istep = 0;
    size_t jstep = 0;
    int i = 0, j = 0;

    HvValArray va;

    assert( idx != NULL );
    assert( val != NULL );

    idxtype = CV_MAT_TYPE( idx->type );
    assert( idxtype == CV_16SC1 || idxtype == CV_32SC1 || idxtype == CV_32FC1 ); // CV_16SC1
    assert( CV_MAT_TYPE( val->type ) == CV_32FC1 ); // CV_32FC1

    istep = val->step;					// 800 = 4(byte)*200(samples:cols)
    jstep = CV_ELEM_SIZE( val->type );	// 4 (float:4 bytes)

    va.data = val->data.ptr; // val의 pointer를 받음.
    va.step = jstep;		 // 4

    for( i = 0; i < sortrows; i++ )	// feature 수
    {
        for( j = 0; j < sortcols; j++ )// training image 수
        {
            CV_MAT_ELEM( *idx, int, i, j ) = (int) j; // 모든 column에 0,1,자연수 넣음.
        }

		SortIndexedValArray_32s( (int*) (idx->data.ptr + i * idx->step), sortcols, &va );

        va.data += istep; // 800
    }
}

/*
int FindStumpThreshold_sq_16s(                                             
        uchar* data,		// valcache의 pointer (특정 feature에 대해서)
		size_t datastep,	// 4 (float)                                         
        uchar* wdata,		// weight의 pointer
		size_t wstep,       // 4 (float)                                   
        uchar* ydata,		// class의 pointer (1.0 or -1.0)
		size_t ystep,		// 4 (float)                                          
        uchar* idxdata,		// sorted index data의 pointer (특정 feature에 대해서)
		size_t idxstep,		// 2 (short)
		int num,			// 200 : m = image 수                           
        float* lerror,		// float* (output)                                                               
        float* rerror,		// float* (output)                                                             
        float* threshold,	// float* (output)
		float* left,		// float* (output)
		float* right,		// float* (output)                             
        float* sumw,		// float* (output)
		float* sumwy,		// float* (output)
		float* sumwyy )		// float* (output)                               
{                                                                                        
    int found = 0;                                                                       
    float wyl  = 0.0F;                                                                   
    float wl   = 0.0F;                                                                   
    float wyyl = 0.0F;                                                                   
    float wyr  = 0.0F;                                                                   
    float wr   = 0.0F;                                                                   
                                                                                         
    float curleft  = 0.0F;                                                               
    float curright = 0.0F;                                                               
    float* prevval = NULL;                                                               
    float* curval  = NULL;                                                               
    float curlerror = 0.0F;                                                              
    float currerror = 0.0F;                                                              
    float wposl;                                                                         
    float wposr;                                                                         
                                                                                         
    int i = 0;                                                                           
    int idx = 0;                                                                         
                                                                                         
    wposl = wposr = 0.0F;                                                                
    if( *sumw == FLT_MAX )                                                               
    {                                                                                    
		// calculate sums                                                              
        float *y = NULL;                                                                 
        float *w = NULL;                                                                 
        float wy = 0.0F;                                                                 
                                                                                         
        *sumw   = 0.0F;                                                                  
        *sumwy  = 0.0F;                                                                  
        *sumwyy = 0.0F;                                                                  
        for( i = 0; i < num; i++ )                                                       
        {                                                                                
            idx = (int) ( *((short*) (idxdata + i*idxstep)) );	// k = idx[i]                            
            w = (float*) (wdata + idx * wstep);					// weight[k]                                      
            *sumw += *w;										// sum_k{ weight[k] }                           
            y = (float*) (ydata + idx * ystep);					//                               
            wy = (*w) * (*y);                                                            
            *sumwy += wy;                                                                
            *sumwyy += wy * (*y);                                                        
        }                                                                                
    }                                                                                    
               
	
    for( i = 0; i < num; i++ ) // image 수                                                           
    {                                                                                    
        idx = (int) ( *((short*) (idxdata + i*idxstep)) );                                
        curval = (float*) (data + idx * datastep);                                       
         // for debug purpose 
    //    if( i > 0 ) assert( (*prevval) <= (*curval) );                                   
                                                                                         
        wyr  = *sumwy - wyl;                                                             
        wr   = *sumw  - wl;                                                              
                                                                                         
        if( wl > 0.0 ) curleft = wyl / wl;                                               
        else curleft = 0.0F;                                                             
                                                                                         
        if( wr > 0.0 ) curright = wyr / wr;                                              
        else curright = 0.0F;                                                            
                                                                                         
        // calculate error (sum of squares)          
        // err = sum( w * (y - left(rigt)Val)^2 )                                      
		// curlerror =(wyyl*wl - wyl*wyl)/wl;
        curlerror = wyyl + curleft * curleft * wl - 2.0F * curleft * wyl;     

        currerror = (*sumwyy) - wyyl + curright * curright * wr - 2.0F * curright * wyr; 
                                                                                         
        if( curlerror + currerror < (*lerror) + (*rerror) )                              
        {                                                                                
            (*lerror) = curlerror;                                                       
            (*rerror) = currerror;                                                       
            *threshold = *curval;                                                        
            if( i > 0 ) {                                                                
                *threshold = 0.5F * (*threshold + *prevval);                             
            }                                                                            
            *left  = curleft;                                                            
            *right = curright;                                                           
            found = 1;                                                                   
        }                                                                                
                                                                                         
        do                                                                               
        {                                                                                
            wl  += *((float*) (wdata + idx * wstep));                                    
            wyl += (*((float*) (wdata + idx * wstep)))                                   
                * (*((float*) (ydata + idx * ystep)));                                   
            wyyl += *((float*) (wdata + idx * wstep))                                    
                * (*((float*) (ydata + idx * ystep)))                                    
                * (*((float*) (ydata + idx * ystep)));                                   

			i++;
			idx = (int) ( *((short*) (idxdata + i*idxstep)));
        }                                                                                
        while( (i < num) &&                                                            
            (*((float*) (data + idx* datastep)) == *curval ) );                                                          
        --i;                                                                             
        prevval = curval;                                                                
    } // for each value 
                                                                                         
    return found;                                                                        
}
*/

int General_FindStumpThreshold_sq_function_32int(                                            
        uchar* data, 
		size_t datastep, 
        uchar* wdata,
		size_t wstep,
        uchar* ydata,
		size_t ystep, 
        uchar* idxdata,
		size_t idxstep, 
		int num,                                         
        float* lerror,                                                                   
        float* rerror,                                                                   
        float* threshold, 
		float* left,
		float* right,                                     
        float* sumw,
		float* sumwy,
		float* sumwyy)
{            
	int found = 0;                                                                       
    float wyl  = 0.0F;                                                                   
    float wl   = 0.0F;                                                                   
    float wyyl = 0.0F;                                                                   
    float wyr  = 0.0F;                                                                   
    float wr   = 0.0F;                                                                   
                                                                                         
    float curleft  = 0.0F;                                                               
    float curright = 0.0F;                                                               
    float* prevval = NULL;                                                               
    float* curval  = NULL;                                                               
    float curlerror = 0.0F;                                                              
    float currerror = 0.0F;                                                              
    float wposl;                                                                         
    float wposr;                                                                         
                                                                                         
    int i = 0;                                                                           
    int idx = 0;                                                                         

	float y,w;
    float wy = 0.0F;                                                                 
	float pos_error = 0.0F;
	float neg_error = 0.0F;

	float sumw_pos = 0.5;
	float sumw_neg = 0.5;
	float sumw_left_pos = 0.0F;
	float sumw_left_neg = 0.0F;

    wposl = wposr = 0.0F;                                                                
    if( *sumw == FLT_MAX )                                                               
    {                                                                                    
        /* calculate sums */                                                             
                                                                                         
        *sumw   = 0.0F;                                                                  
        *sumwy  = 0.0F;                                                                  
        *sumwyy = 0.0F;  
		sumw_pos = 0.0F;
		sumw_neg = 0.0F;
        for( i = 0; i < num; i++ )                                                       
        {                                                                                
            idx = (int) ( *((int*) (idxdata + i*idxstep)) );                            
            w = *(float*) (wdata + idx * wstep);                                          
            y = *(float*) (ydata + idx * ystep);

            *sumw += w;                                                                 
			if(y>0)
				sumw_pos += w;
			else sumw_neg += w;

            wy = w * y;                                                            
            *sumwy += wy;                                                                
            *sumwyy += wy * y;                                                        
        }                                                                                
    }      
	
           
    for( i = 0; i < num; i++ )                                                           
    {                                                                                    
        idx = (int) ( *((int*) (idxdata + i*idxstep)) );  
        curval = (float*) (data + idx * datastep); 
 /*
		y = *(float*) (ydata + idx * ystep);
	    w = *((float*) (wdata + idx * wstep)); 
                                                                                     
 	    if(y>0)
		    sumw_left_pos += w;
	    else sumw_left_neg += w;
*/
		wyr  = *sumwy - wyl; 
        wr   = *sumw  - wl; 
                                                                                         
        if(wl > 0.0 ) curleft = wyl / wl; 
        else curleft = 0.0F;
                                                                                        
        if(wr > 0.0 ) curright = wyr / wr;
        else curright = 0.0F; 

		if(curleft < curright)
		{ // left is neg side
			curlerror = sumw_left_pos;
			currerror = sumw_neg - sumw_left_neg;
		}
		else
		{
			curlerror = sumw_left_neg;
			currerror = sumw_pos - sumw_left_pos;
		}
                                                                                        
        
                                                                                        
        if((curlerror+currerror<(*lerror)+(*rerror)) && 
		   (curlerror>=0.0f) && (currerror>=0.0f) )
        {                                                                               
            (*lerror) = curlerror;                                                      
            (*rerror) = currerror;                                                      
            *threshold = *curval;                                                       
            if( i > 0 ) {                                                               
                *threshold = 0.5F * (*threshold + *prevval);                            
            }                                                                           
            *left  = curleft;                                                           
            *right = curright;                                                          
            found = 1;                                                                  
        }                                                                               
                                                                                        
        do                                                                              
        {             
           y = *(float*) (ydata + idx * ystep);
		   w = *((float*) (wdata + idx * wstep)); 
		   wl  += w;
		   wyl += w*y;
		   wyyl += w*y*y;
		   
		   if(y>0)
			   sumw_left_pos += w;
		   else sumw_left_neg += w;
				
        }                                                                               
        while( (++i) < num &&                                                           
            ( *((float*) (data + (idx =                                                 
                (int) ( *((int*) (idxdata + i*idxstep))) ) * datastep))                
                == *curval ) );                                                         
        --i;                                                                            
        prevval = curval;                                                               
    } /* for each value */                                                              
                     
    return found;                                                                       
}

int FindStumpThreshold_sq_function_32int(                                            
        uchar* data, 
		size_t datastep, 
        uchar* wdata,
		size_t wstep,
        uchar* ydata,
		size_t ystep, 
        uchar* idxdata,
		size_t idxstep, 
		int num,                                         
        float* lerror,                                                                   
        float* rerror,                                                                   
        float* threshold, 
		float* left,
		float* right,                                     
        float* sumw,
		float* sumwy,
		float* sumwyy )                                       
{            
	int found = 0;                                                                       
    float wyl  = 0.0F;                                                                   
    float wl   = 0.0F;                                                                   
    float wyyl = 0.0F;                                                                   
    float wyr  = 0.0F;                                                                   
    float wr   = 0.0F;                                                                   
                                                                                         
    float curleft  = 0.0F;                                                               
    float curright = 0.0F;                                                               
    float* prevval = NULL;                                                               
    float* curval  = NULL;                                                               
    float curlerror = 0.0F;                                                              
    float currerror = 0.0F;                                                              
    float wposl;                                                                         
    float wposr;                                                                         
                                                                                         
    int i = 0;                                                                           
    int idx = 0;                                                                         
                                                                                         
    wposl = wposr = 0.0F;                                                                
    if( *sumw == FLT_MAX )                                                               
    {                                                                                    
        /* calculate sums */                                                             
        float *y = NULL;                                                                 
        float *w = NULL;                                                                 
        float wy = 0.0F;                                                                 
                                                                                         
        *sumw   = 0.0F;                                                                  
        *sumwy  = 0.0F;                                                                  
        *sumwyy = 0.0F;                                                                  
        for( i = 0; i < num; i++ )                                                       
        {                                                                                
            idx = (int) ( *((int*) (idxdata + i*idxstep)) );                            
            w = (float*) (wdata + idx * wstep);                                          
            *sumw += *w;                                                                 
            y = (float*) (ydata + idx * ystep);                                          
            wy = (*w) * (*y);                                                            
            *sumwy += wy;                                                                
            *sumwyy += wy * (*y);                                                        
        }                                                                                
    }                                                                                    
                                                                                         
    for( i = 0; i < num; i++ )                                                           
    {                                                                                    
        idx = (int) ( *((int*) (idxdata + i*idxstep)) );                                
        curval = (float*) (data + idx * datastep);                                       
         /* for debug purpose */                                                         
//        if( i > 0 ) assert( (*prevval) <= (*curval) );                                   
                                                                                         
        wyr  = *sumwy - wyl;                                                             
        wr   = *sumw  - wl;                                                              
                                                                                         
        if( wl > 0.0 ) curleft = wyl / wl;                                               
        else curleft = 0.0F;                                                            
                                                                                        
        if( wr > 0.0 ) curright = wyr / wr;                                             
        else curright = 0.0F;                                                           
                                                                                        
        /* calculate error (sum of squares)          */                                 
        /* err = sum( w * (y - left(rigt)Val)^2 )    */                                 
        curlerror = wyyl + curleft * curleft * wl - 2.0F * curleft * wyl;  
/*
		float ftemp1 = wyyl - curleft * curleft * wl;
		if(ftemp1 - curlerror> 0.000001)
		{
			int mmmmm=0;
		}
*/
        currerror = (*sumwyy) - wyyl + curright * curright * wr - 2.0F * curright * wyr;
                                                                                        
        if((curlerror+currerror<(*lerror)+(*rerror)) && 
		   (curlerror>=0.0f) && (currerror>=0.0f) )
        {                                                                               
            (*lerror) = curlerror;                                                      
            (*rerror) = currerror;                                                      
            *threshold = *curval;                                                       
            if( i > 0 ) {                                                               
                *threshold = 0.5F * (*threshold + *prevval);                            
            }                                                                           
            *left  = curleft;                                                           
            *right = curright;                                                          
            found = 1;                                                                  
        }                                                                               
                                                                                        
        do                                                                              
        {                                                                               
            wl  += *((float*) (wdata + idx * wstep));                                   
            wyl += (*((float*) (wdata + idx * wstep)))                                  
                * (*((float*) (ydata + idx * ystep)));                                  
            wyyl += *((float*) (wdata + idx * wstep))                                   
                * (*((float*) (ydata + idx * ystep)))                                   
                * (*((float*) (ydata + idx * ystep)));                                  
        }                                                                               
        while( (++i) < num &&                                                           
            ( *((float*) (data + (idx =                                                 
                (int) ( *((int*) (idxdata + i*idxstep))) ) * datastep))                
                == *curval ) );                                                         
        --i;                                                                            
        prevval = curval;                                                               
    } /* for each value */                                                              
                     
    return found;                                                                       
}

float CalRealAdaBoosting_Epsilon(int nBinNum)
{
	float Epsilon = 0.01/nBinNum;
	return Epsilon;
}

bool FindRealStump_function_32int(                                            
        uchar* data, 
		size_t datastep, 
        uchar* wdata,
		size_t wstep,
        uchar* ydata,
		size_t ystep, 
        uchar* idxdata,
		size_t idxstep, 
		int num,                                         
        float* fError,
		int nBinNum,
        float* fMin,
		float* fBinWidth_Inv,
		float* fVal
		)                                       
{  
	float Epsilon = CalRealAdaBoosting_Epsilon(nBinNum);

	float fBinPosNum[MAX_REAL_CLASSIFIER_BIN], fBinNegNum[MAX_REAL_CLASSIFIER_BIN];
	int i;

	/// set small number to get robust result
	for(i=0;i<MAX_REAL_CLASSIFIER_BIN;i++)
	{
		fBinPosNum[i] = 0.0f;
		fBinNegNum[i] = 0.0f;
	}
    int idx = (int) ( *((int*) idxdata)); 
    float fLocal_MinValue = *((float*) (data+idx * datastep));
    idx = (int) ( *((int*) (idxdata + (num-1)*idxstep)));
    float fLocal_MaxValue = *((float*) (data + idx * datastep));
	float fLocal_BinWidth_Inv = nBinNum/(fLocal_MaxValue - fLocal_MinValue);

	float curval,y,w;
	int nBinNo;
	/// count bin sample number
	for(i=0;i<num;i++)
	{
        idx = (int) ( *((int*) (idxdata + i*idxstep)) );                                
        curval = *((float*) (data + idx * datastep));                                       
        y = *((float*) (ydata + idx * ystep));                                          
        w = *((float*) (wdata + idx * wstep));
		nBinNo = int((curval - fLocal_MinValue) * fLocal_BinWidth_Inv);
		
		if(nBinNo>=nBinNum)
			nBinNo = nBinNum - 1;

		if(y>0) 
			fBinPosNum[nBinNo] += w;
		else fBinNegNum[nBinNo] += w;
	}

	/// get local sample distribution
	float fLocalError = 0;
	for(i=0;i<nBinNum;i++)
		fLocalError += sqrt(fBinPosNum[i]*fBinNegNum[i]);

	if(fLocalError < *fError)
	{
		float fTemp;
		*fError = fLocalError;
		for(i=0;i<nBinNum;i++)
		{
			fTemp = (fBinPosNum[i] + Epsilon)/(fBinNegNum[i] + Epsilon);
			fVal[i] = 0.5*log(fTemp);
		}

		*fBinWidth_Inv = fLocal_BinWidth_Inv;
		*fMin = fLocal_MinValue;

		return true;
	}
	/// calculate the error

    return false;                                                                       
}

void ConvertToFastFeature( HvFeature* feature,
                           HvFastFeature* fastFeature,
                           int size, int step,
						   double dScale)
{
    int i = 0, j = 0;

	CvRect Scale_Rect;
	int dScale_step = step* dScale;
    for( i = 0; i < size; i++ )
    {
        fastFeature[i].tilted = feature[i].tilted; // tilted는 그대로.

        if( !fastFeature[i].tilted ) // not tilted
        {
            for( j = 0; j < HV_FEATURE_MAX; j++ )
            {
                fastFeature[i].rect[j].weight = feature[i].rect[j].weight; // weight는 그대로.
                if( fastFeature[i].rect[j].weight == 0.0F )
                {
                    break;
                }
				Scale_Rect.x = int(feature[i].rect[j].r.x * dScale);
				Scale_Rect.y = int(feature[i].rect[j].r.y * dScale);
				Scale_Rect.width = int(feature[i].rect[j].r.width * dScale);
				Scale_Rect.height = int(feature[i].rect[j].r.height * dScale);

                HV_SUM_OFFSETS_Func( &(fastFeature[i].rect[j].p0),
                                &(fastFeature[i].rect[j].p1),
                                &(fastFeature[i].rect[j].p2),
                                &(fastFeature[i].rect[j].p3),
                                Scale_Rect, dScale_step);
            }
            
        }
        else	// tilted
        {
            for( j = 0; j < HV_FEATURE_MAX; j++ )
            {
                fastFeature[i].rect[j].weight = feature[i].rect[j].weight; // weight는 그대로.
                if( fastFeature[i].rect[j].weight == 0.0F )
                {
                    break;
                }
                HV_TILTED_OFFSETS( fastFeature[i].rect[j].p0,
                                   fastFeature[i].rect[j].p1,
                                   fastFeature[i].rect[j].p2,
                                   fastFeature[i].rect[j].p3,
                                   feature[i].rect[j].r, step )
            }
        }
    }
}

void ConvertToFastFeature( HvFeature* feature,
                           HvFastFeature* fastFeature,
                           int size, int step,
						   int x, int y,
						   int dScale)
{
    int i = 0, j = 0;

	CvRect Scale_Rect;
    for( i = 0; i < size; i++ )
    {
        fastFeature[i].tilted = feature[i].tilted; // tilted는 그대로.

        if( !fastFeature[i].tilted ) // not tilted
        {
            for( j = 0; j < HV_FEATURE_MAX; j++ )
            {
                fastFeature[i].rect[j].weight = feature[i].rect[j].weight; // weight는 그대로.
                if( fastFeature[i].rect[j].weight == 0.0F )
                {
                    break;
                }
				Scale_Rect.x = int(feature[i].rect[j].r.x * dScale + x + 0.5);
				Scale_Rect.y = int(feature[i].rect[j].r.y * dScale + y + 0.5);
				Scale_Rect.width = int(feature[i].rect[j].r.width * dScale);
				Scale_Rect.height = int(feature[i].rect[j].r.height * dScale);

                HV_SUM_OFFSETS_Func( &(fastFeature[i].rect[j].p0),
                                &(fastFeature[i].rect[j].p1),
                                &(fastFeature[i].rect[j].p2),
                                &(fastFeature[i].rect[j].p3),
                                Scale_Rect, step);
            }
            
        }
        else	// tilted
        {
            for( j = 0; j < HV_FEATURE_MAX; j++ )
            {
                fastFeature[i].rect[j].weight = feature[i].rect[j].weight; // weight는 그대로.
                if( fastFeature[i].rect[j].weight == 0.0F )
                {
                    break;
                }
                HV_TILTED_OFFSETS( fastFeature[i].rect[j].p0,
                                   fastFeature[i].rect[j].p1,
                                   fastFeature[i].rect[j].p2,
                                   fastFeature[i].rect[j].p3,
                                   feature[i].rect[j].r, step )
            }
        }
    }
}

void GetAuxImages(CvMat* img, CvMat* sum, CvMat* tilted,
                  CvMat* sqsum, float *mean, float* normfactor,
				  int nOrignalWidth, int nOrignalHeight )
{
    CvRect normrect;
    int p0, p1, p2, p3;
    double   valsum   = 0.0;
    double valsqsum = 0;
    double area = 0.0;
    
	double dScale = img->width *1.0/ nOrignalWidth;

    cvIntegral( img, sum, sqsum, tilted );  // Integral Image. 나중에 함수 따로 뽑아내어야 함. 
    normrect = cvRect( int(dScale)*1, int(dScale)*1, int(dScale*(nOrignalWidth- 2)), int(dScale*(nOrignalHeight-2)) );	// normrect: 4개의 경계에서 1 pixel씩을 제외한다.  
	
    HV_SUM_OFFSETS( p0, p1, p2, p3, normrect, img->cols + 1 )	// normrect에 맞게 p0, p1, p2, p3를 만든다.
    
    area = normrect.width * normrect.height;
    valsum = ((int*) (sum->data.ptr))[p0] - ((int*) (sum->data.ptr))[p1]
           - ((int*) (sum->data.ptr))[p2] + ((int*) (sum->data.ptr))[p3];
    valsqsum = ((double*) (sqsum->data.ptr))[p0]
             - ((double*) (sqsum->data.ptr))[p1]
             - ((double*) (sqsum->data.ptr))[p2]
             + ((double*) (sqsum->data.ptr))[p3];

	*mean = valsum/ area;
	// normfactor : N*sigma = total number X standard deviation
	(*normfactor) = (float) sqrt( (double) (area * valsqsum - valsum * valsum) );
}

void biGammaCorrection(IplImage* IPl_Image)
{
    int nChannel = IPl_Image->nChannels;
    if (nChannel != 1)
        return;

    unsigned char Gamma[] = {
        0x00, 0x15, 0x1d, 0x22, 0x27, 0x2b, 0x2f, 0x32,
        0x36, 0x39, 0x3b, 0x3e, 0x40, 0x43, 0x45, 0x47,
        0x49, 0x4b, 0x4d, 0x4f, 0x51, 0x53, 0x55, 0x56,
        0x58, 0x5a, 0x5b, 0x5d, 0x5e, 0x60, 0x61, 0x63,
        0x64, 0x65, 0x67, 0x68, 0x69, 0x6b, 0x6c, 0x6d,
        0x6f, 0x70, 0x71, 0x72, 0x73, 0x75, 0x76, 0x77,
        0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7e, 0x7f, 0x80,
        0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
        0x89, 0x8a, 0x8b, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
        0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0x9a, 0x9b, 0x9c, 0x9d,
        0x9e, 0x9f, 0x9f, 0xa0, 0xa1, 0xa2, 0xa2, 0xa3,
        0xa4, 0xa5, 0xa6, 0xa6, 0xa7, 0xa8, 0xa9, 0xa9,
        0xaa, 0xab, 0xab, 0xac, 0xad, 0xae, 0xae, 0xaf,
        0xb0, 0xb0, 0xb1, 0xb2, 0xb3, 0xb3, 0xb4, 0xb5,
        0xb5, 0xb6, 0xb7, 0xb7, 0xb8, 0xb9, 0xb9, 0xba,
        0xbb, 0xbb, 0xbc, 0xbd, 0xbd, 0xbe, 0xbf, 0xbf,
        0xc0, 0xc0, 0xc1, 0xc2, 0xc2, 0xc3, 0xc4, 0xc4,
        0xc5, 0xc5, 0xc6, 0xc7, 0xc7, 0xc8, 0xc8, 0xc9,
        0xca, 0xca, 0xcb, 0xcb, 0xcc, 0xcd, 0xcd, 0xce,
        0xce, 0xcf, 0xd0, 0xd0, 0xd1, 0xd1, 0xd2, 0xd2,
        0xd3, 0xd4, 0xd4, 0xd5, 0xd5, 0xd6, 0xd6, 0xd7,
        0xd7, 0xd8, 0xd9, 0xd9, 0xda, 0xda, 0xdb, 0xdb,
        0xdc, 0xdc, 0xdd, 0xdd, 0xde, 0xde, 0xdf, 0xe0,
        0xe0, 0xe1, 0xe1, 0xe2, 0xe2, 0xe3, 0xe3, 0xe4,
        0xe4, 0xe5, 0xe5, 0xe6, 0xe6, 0xe7, 0xe7, 0xe8,
        0xe8, 0xe9, 0xe9, 0xea, 0xea, 0xeb, 0xeb, 0xec,
        0xec, 0xed, 0xed, 0xee, 0xee, 0xef, 0xef, 0xf0,
        0xf0, 0xf1, 0xf1, 0xf2, 0xf2, 0xf3, 0xf3, 0xf3,
        0xf4, 0xf4, 0xf5, 0xf5, 0xf6, 0xf6, 0xf7, 0xf7,
        0xf8, 0xf8, 0xf9, 0xf9, 0xfa, 0xfa, 0xfa, 0xfb,
        0xfb, 0xfc, 0xfc, 0xfd, 0xfd, 0xfe, 0xfe, 0xff
    };

    int i, j;
    int width = IPl_Image->width;
    int height = IPl_Image->height;

    unsigned char *lpImage = (unsigned char *)(IPl_Image->imageData);
    int nWidthStep = IPl_Image->widthStep;
    unsigned char pixel;
    for (i = 0; i <height; i++)
    {
        for (j = 0; j <width; j++)
        {
            pixel = *(lpImage + j);
            *(lpImage + j) = Gamma[pixel];
        }
        lpImage += nWidthStep;
    }
}
