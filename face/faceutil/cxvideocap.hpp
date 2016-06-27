/**
*** Copyright (C) 1985-2011 Intel Corporation.  All rights reserved.
***
*** The information and source code contained herein is the exclusive
*** property of Intel Corporation and may not be disclosed, examined
*** or reproduced in whole or in part without explicit written authorization
*** from the company.
***
*** Embedded Application Lab, Intel Labs China.
**/

#ifndef _CX_VIDEOCAP_HPP_
#define _CX_VIDEOCAP_HPP_

#include <opencv/highgui.h>

#pragma warning ( disable: 4996 )

class CxImageSeqReader
{
public:
    CxImageSeqReader() {}
    virtual ~CxImageSeqReader() {};

    virtual IplImage* query() = 0;
    virtual bool eof() = 0;
    virtual int width() const = 0;
    virtual int width( int value ) { return 0; }
    virtual int height() const = 0;
    virtual int height( int value ) { return 0; }
    virtual double fps() const { return 0; }
    virtual double fps( double value ) { return 0; }
    virtual int index() const = 0;
    virtual const char* filename() const = 0;
};

class CxPicListReader : public CxImageSeqReader
{
public:
    // constructor
    CxPicListReader( const char *lst_file ) 
    {
        ptr_image = NULL;
        num_lines = 0;
        ptr_file = fopen( lst_file, "r" );
        if( ptr_file == NULL )
            OPENCV_ERROR( CV_StsBadArg, 
                          "CxPicListReader::CxPicListReader()", 
                          "Cannot open file list." );
    }

    // destructor
    virtual ~CxPicListReader() 
    { 
        if( ptr_image )
            cvReleaseImage( &ptr_image );
        if( ptr_file )
            fclose( ptr_file );
    }

    // query
    virtual IplImage* query() 
    { 
        int flags=CV_LOAD_IMAGE_COLOR;
        if( feof(ptr_file) )
            return NULL;

	   char* ptr = fgets(pic_name, 1024, ptr_file);
       //if( fscanf( ptr_file, "%s", pic_name ) == 1 )
	   if(ptr)
	   {
		   int len = (int)strlen(pic_name);
		   if(pic_name[len-1] == 10) pic_name[len-1] = 0;
		   num_lines ++;
	   }
        else
        {
            pic_name[0] = '\0';
            return NULL;
        }
        // release old image
        if( ptr_image )
            cvReleaseImage( &ptr_image );
        // load new image
        ptr_image = cvLoadImage( pic_name, flags );
        return ptr_image;
    }

    // properties
    virtual bool eof() { return feof(ptr_file) != 0; }

    virtual int fps() { return 0; }

    virtual int width() const { return ptr_image ? ptr_image->width : 0; }
    
    virtual int height() const { return ptr_image ? ptr_image->height : 0; }
    
    virtual int index() const { return num_lines; }

    virtual const char* filename() const { return pic_name; }
    
protected:
    FILE*       ptr_file;
    char        pic_name[1024];
    IplImage*   ptr_image;
    int         num_lines;

};


class CxVideoReader : public CxImageSeqReader
{
public:
    // constructor
    CxVideoReader( const char *filename ) 
    {
        if( filename[0]>='0' && filename[0]<='9' && filename[1]=='\0' )
            capture = cvCreateCameraCapture( filename[0] - '0' );
        else
            capture = cvCreateFileCapture( filename );
        if( ! capture )
        {
            OPENCV_ERROR( CV_StsBadArg, 
                          "CxVideoReader::CxVideoReader()", 
                          "No video source." );
            return;
        }

		sprintf( pic_name, "%s", filename);
    }

    // destructor
    virtual ~CxVideoReader() { cvReleaseCapture( &capture ); }

    // query
    virtual IplImage* query() 
    {
        IplImage* image = cvQueryFrame( capture ); 
        //sprintf( pic_name, "Frame#%06d", index() );
        return image;
    }

    virtual bool eof() { return (count() > 0) && (index() >= count()); }

    // properties
    virtual int width() const
    { return (int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH ); }
    
    virtual int width( int value ) 
    { return cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, value ); }

    virtual int height() const
    { return (int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT ); }
    
    virtual int height( int value ) 
    { return cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, value ); }

    int count() const
    { return (int)cvGetCaptureProperty( capture, CV_CAP_PROP_FRAME_COUNT ); }
    
    virtual int index() const
    { return (int)cvGetCaptureProperty( capture, CV_CAP_PROP_POS_FRAMES ); }
    
    int index( int value )
    { return cvSetCaptureProperty( capture, CV_CAP_PROP_POS_FRAMES, value ); }

    virtual double fps() const  
    { return cvGetCaptureProperty( capture, CV_CAP_PROP_FPS ); }

    virtual double fps( double value ) 
    { return cvSetCaptureProperty( capture, CV_CAP_PROP_FPS, value ); }

    int msec() const 
    { return (int)cvGetCaptureProperty( capture, CV_CAP_PROP_POS_MSEC ); }

    int msec( int value )
    { return cvSetCaptureProperty( capture, CV_CAP_PROP_POS_MSEC, value ); }
 
    void config( int _width, int _height, double _fps )
    {
        width( _width );
        height( _height );
        fps( _fps );
    }

    virtual const char* filename() const { return pic_name; }

 
    // operator
    operator const CvCapture* () const { return capture; }

protected:
    CvCapture*  capture;
    char        pic_name[1024];

};

class CxVideoWriter 
{
public:
    // constructor
    CxVideoWriter( const char* filename, double fps=30, int fourcc=-1 ) 
    {
        char* q = this->filename;
        while( (*q++ = *filename++) );

        this->fps = (fps<=0) ? 30 : fps;
        this->fourcc = fourcc;
        writer = NULL;
    }

    // destructor
    virtual ~CxVideoWriter() 
    {
        if( writer )
            cvReleaseVideoWriter( &writer );
    }

    // write frame
    void write( const IplImage* image ) 
    {
        OPENCV_ASSERT( image->width % 4 == 0, 
                       "CxVideoWriter::write()", 
                       "The width must be multiple of 4." );
        if( writer==NULL ) 
        {
            writer = cvCreateVideoWriter( filename, fourcc, fps, 
                    cvGetSize(image), 1 );
            OPENCV_ASSERT( writer, 
                    "CxVideoWriter::write)", 
                    "Failed to create video.");
        }
        cvWriteFrame( writer, image );
    }

    // operator
    operator const CvVideoWriter* () const { return writer; }
    operator CvVideoWriter* () { return writer; }

protected:
    char    filename[1024];
    double  fps;
    int     fourcc;
    CvVideoWriter* writer;

};

#endif // _CX_VIDEOCAP_HPP_
