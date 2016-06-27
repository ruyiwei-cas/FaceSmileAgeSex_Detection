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

#include <stdio.h>
#include <direct.h>
#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "cxvideocap.hpp"
#include "cxlibface.hpp"

#pragma comment ( lib, "cxcore200.lib")
#pragma comment ( lib, "cv200.lib")
#pragma comment ( lib, "highgui200.lib")
#pragma comment ( lib, "ml200.lib")

#pragma comment ( lib, "facelib.lib")

#pragma warning ( disable: 4996 )
#pragma warning ( disable: 4305 )
#pragma warning ( disable: 4244 )

#ifndef WIN32
#define stricmp strcasecmp
#endif

// add a new faceset for face recognition
void addFaceSet(int x, int y, void* param)
{   
	CxMouseParam *mouse_param = (CxMouseParam*)param;
	IplImage* img = (IplImage*) mouse_param->image;

	//stop video play and input face name
	bool   bSel = false;
	int    id_cur;
	CvRect rect_cur;
	for(int i = 0; i < mouse_param->face_num; i++)
	{
		rect_cur = mouse_param->rects[i].rc;
		if(x > rect_cur.x && y > rect_cur.y && x < rect_cur.x+rect_cur.width && y < rect_cur.y+rect_cur.height)
		{
			bSel   = true;
			id_cur = i;
			break;
		}
	}

	if(bSel == true)
	{
		mouse_param->play = false; //stop video play
		int facetrack_id= mouse_param->rects[id_cur].fid;
		mouse_param->ret_facetrack_id = facetrack_id;

		char sCaption[256];
		CvFont *pFont = new CvFont;
		cvInitFont(pFont, CV_FONT_HERSHEY_PLAIN, 0.85, 0.85, 0, 1);

		sprintf(sCaption, "Insert a new face for track_id = %d. Pls input name in the cmd window", facetrack_id);
		printf("%s\n", sCaption);
		cxlibDrawCaption( mouse_param->image, pFont, sCaption);
		cvShowImage( "Face Tester",  mouse_param->image );
		cvWaitKey( 30 );

		int nFaceSetIdx    = -1;
		int nFaceSetID     = -1;
		float probFaceID  = 0;
		char sFaceName[32] = "";
		CxlibFaceAnalyzer   *pFaceAnalyzer;
		CxlibFaceRecognizer *pFaceRecognizer;
		if(mouse_param->typeRecognizer == 0)
		{
			pFaceAnalyzer = (CxlibFaceAnalyzer*)mouse_param->faceRecognizer;
			nFaceSetID    =  pFaceAnalyzer->predictFaceSet(mouse_param->cut_big_face, &probFaceID, sFaceName);
		}
		else
		{
			pFaceRecognizer = (CxlibFaceRecognizer*)mouse_param->faceRecognizer;
			nFaceSetID = pFaceRecognizer->predict(mouse_param->cut_big_face, &probFaceID);
			const char *name = pFaceRecognizer->getFaceName(nFaceSetID);
			strcpy(sFaceName, name);
		}

		//input new name
		char nameBuff[32] ="";
		char *name = NULL;
		printf("Old name:[%s]; New name:", sFaceName);
		name = gets( nameBuff );

		if(name[0] != 0)
		{   
			//add a new name
			if(mouse_param->typeRecognizer == 0)
			{
				nFaceSetIdx    =  pFaceAnalyzer->insertEmptyFaceSet(name);
				pFaceAnalyzer->tryInsertFace(mouse_param->cut_big_face, nFaceSetIdx);
				nFaceSetID     =  pFaceAnalyzer->getFaceSetID(nFaceSetIdx);
			}
			else
			{
				nFaceSetIdx = pFaceRecognizer->insertEmptyFaceSet(name);
				pFaceRecognizer->tryInsertFace(mouse_param->cut_big_face, nFaceSetIdx);
				nFaceSetID  =  pFaceRecognizer->getFaceSetID(nFaceSetIdx);
			}

			
			mouse_param->updated = true;
			mouse_param->ret_online_collecting = 1; //enable online face exemplar collecting

			sprintf(sCaption, "set as the new name [%s] for track_id = %d\n", name, facetrack_id);
			printf("%s\n", sCaption);
			cxlibDrawCaption( mouse_param->image, pFont, sCaption);
			cvShowImage( "Face Tester",  mouse_param->image );
			cvWaitKey( 30 );
		}
		else
		{
			sprintf(sCaption, "Cancel, still set as the old name [%s] for track_id = %d\n", sFaceName, facetrack_id);
			printf("%s\n", sCaption);
			cxlibDrawCaption( mouse_param->image, pFont, sCaption);
			cvShowImage( "Face Tester",  mouse_param->image );
			cvWaitKey( 30 );
		}

		mouse_param->ret_faceset_id = nFaceSetID;
		mouse_param->play = true;

		delete pFont;
	}
}

void my_mouse_callback( int event, int x, int y, int flags, void* param )
{
	switch( event )
	{
	case CV_EVENT_RBUTTONDOWN: 
		break;
	case CV_EVENT_RBUTTONUP: 
		addFaceSet( x, y, param );
		break;
	case CV_EVENT_MOUSEMOVE: 
		break;
	case CV_EVENT_LBUTTONUP: 
		break;
	}
}

int testfaceLib_sThread ( const char* str_video, int  trackerType, int multiviewType, int recognizerType, const char* str_facesetxml, int threads, 
						 bool blink, bool smile, bool gender, bool age, bool recog, bool quiet, bool saveface, const char* sfolder, bool bEnableAutoCluster)
{
	int  faceimgID = 0;
	char driver[8];
	char dir[1024];
	char fname[1024];
	char ext[8];
	char sImgPath[1024];

	if(sfolder)
	{
		char sysCommand[128];
		sprintf (sysCommand, "mkdir %s", sfolder);
		system (sysCommand);

		sprintf(sImgPath, "%s//%s", sfolder,  "imaginfo.txt");
		sprintf(fname,   "%s//%s", sfolder,  "faceinfo.txt");
	}
	else
	{
		sprintf(sImgPath, "%s", "imaginfo.txt");
		sprintf(fname,   "%s", "faceinfo.txt");
	}

	FILE* fp_imaginfo = fopen( sImgPath, "wt" );
    FILE* fp_faceinfo = fopen( fname, "wt" );

    bool bAutoFocus = false;
	IplImage *imgAutoFocus = NULL;

	/////////////////////////////////////////////////////////////////////////////////////
	//	init GUI window
	const char* str_title = "Face Tester";
	if( ! quiet )
		cvNamedWindow( str_title, CV_WINDOW_AUTOSIZE );

	char sCaptionInfo[256]="";
	CvFont *pFont = new CvFont;
	cvInitFont(pFont, CV_FONT_HERSHEY_PLAIN, 0.85, 0.85, 0, 1);
	
	// load GUI smile icon images
	IplImage *pImgSmileBGR;
	IplImage *pImgSmileMask;
	if(age == 0)
	{
		pImgSmileBGR  = cvLoadImage( "smile.bmp" );
		pImgSmileMask = cvLoadImage( "smilemask.bmp", 0 );
	}
	else
	{
		pImgSmileBGR  = cvLoadImage( "faceicon.bmp" );
		pImgSmileMask = cvLoadImage( "faceiconMask.bmp", 0 );
	}
	IplImage *pImgSmileBGRA = cvCreateImage( cvSize(pImgSmileBGR->width, pImgSmileBGR->height), IPL_DEPTH_8U, 4 );
	cvCvtColor(pImgSmileBGR, pImgSmileBGRA, CV_BGR2BGRA );

	// open video source
    size_t len = strlen( str_video );
    bool is_piclist = (0 == stricmp( str_video + len - 4, ".txt" ));
    CxImageSeqReader* vidcap = NULL;
    if( is_piclist )
        vidcap = new CxPicListReader( str_video );
    else
        vidcap = new CxVideoReader( str_video );

	if( cvGetErrStatus() < 0 )
	{   
		cvSetErrStatus( CV_StsOk );
		return -1;
	}

	// when using camera, set to 640x480, 30fps
	if( isdigit(str_video[0]) != 0 && str_video[1] == '\0' )
	{
		vidcap->width( 640 );
		vidcap->height( 480 );
		vidcap->fps( 30 );
	}

	// print beginning info
	printf( "tracker cascade:  '%s'\n", trackerType == TRA_HAAR ? "haar" : (trackerType== TRA_SURF ? "surf" : "pf tracker SURF"));
	printf( "face recognizer:  '%s'\n", recognizerType == RECOGNIZER_BOOST_GB240 ? "boost gabor240" : "cascade gloh"  );
	printf( "video:    '%s', %dx%d, %2.1f fps\n", str_video, 
		vidcap->width(), vidcap->height(), vidcap->fps() );

	// config face tracker
	const int  face_max = 16;
	CvRectItem rects[face_max];
	
	tagDetectConfig configParam;
	EnumViewAngle  viewAngle = (EnumViewAngle)multiviewType;

	CxlibFaceDetector detector;
	detector.init(viewAngle, (EnumFeaType)trackerType);
	detector.config( configParam );

	CxlibFaceTracker tracker;
	tracker.init(viewAngle, (EnumTrackerType)trackerType);
	tracker.config( configParam, TR_NLEVEL_3 );

	if( cvGetErrStatus() < 0 )
	{
		cvSetErrStatus( CV_StsOk );
		return -1;
	}

	// config landmark detector
	CvPoint2D32f   landmark6[6+1]; // consider both 6-pt and 7-pt
	float          parameters[16];
	bool      bLandmark = false;
	CxlibLandmarkDetector landmarkDetector(LDM_6PT);

	int size_smallface = 64;
	int size_bigface   = 128;
	CxlibAlignFace cutFace(size_smallface, size_bigface);
	
	// config blink/smile/gender detector
	int    bBlink = 0, bSmile = 0, bGender = 0, bAge = 0;  //+1, -1, otherwise 0: no process 
	float  probBlink = 0, probSmile = 0, probGender = 0, probAge[4];
	int    nAgeID = 0;

	CxlibBlinkDetector  blinkDetector(size_smallface);
	CxlibSmileDetector  smileDetector(size_smallface);
	CxlibGenderDetector genderDetector(size_smallface);
	CxlibAgeDetector    ageDetector(size_bigface);

	// config face recognizer
	float probFaceID = 0;
	if(str_facesetxml == NULL)
		str_facesetxml = "faceset_model.xml";

	CxlibFaceRecognizer faceRecognizer( size_bigface, recognizerType );
	if(recog) faceRecognizer.loadFaceModelXML(str_facesetxml);
	
	// set mouse event process
	CxMouseParam mouse_faceparam;
	mouse_faceparam.updated = false;
	mouse_faceparam.play = true;
	mouse_faceparam.ret_online_collecting = 0;
		
	if(! quiet)
	{
		mouse_faceparam.face_num  = face_max;
		mouse_faceparam.rects     = rects;
		mouse_faceparam.image     = NULL;
		mouse_faceparam.cut_big_face= cutFace.getBigCutFace();
		mouse_faceparam.typeRecognizer = 1;
		mouse_faceparam.faceRecognizer = &faceRecognizer;
		cvSetMouseCallback(	str_title, my_mouse_callback, (void*)&mouse_faceparam );
	}

	// init count ticks                   
	int64  ticks, start_ticks, total_ticks;
	int64  tracker_total_ticks, landmark_total_ticks, align_total_ticks,
		   blink_total_ticks, smile_total_ticks, gender_total_ticks, age_total_ticks, recg_total_ticks;
	double frame_fps, tracker_fps, landmark_fps, align_fps, blink_fps, smile_fps, gender_fps, age_fps, recg_fps, total_fps; 

	start_ticks         = total_ticks          = 0;
	tracker_total_ticks = landmark_total_ticks = align_total_ticks  = 0;
	blink_total_ticks   = smile_total_ticks    = gender_total_ticks = age_total_ticks = recg_total_ticks = 0;

	tracker_fps = landmark_fps = align_fps = blink_fps = smile_fps = gender_fps = age_fps = recg_fps = total_fps = 0.0;        

	// loop for each frame of a video/camera
	int frames = 0;
	IplImage *pImg = NULL;
	int   print_faceid=-1;
	float print_score = 0;
	std::string  print_facename;

	bool bRunLandmark = blink || smile|| gender|| age|| recog || saveface;
	IplImage *thumbnailImg   = cvCreateImage(cvSize(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), IPL_DEPTH_8U, 3);   
	
	//dynamic clustering for smooth ID registration
	//bEnableAutoCluster = true;
	if( is_piclist ) bEnableAutoCluster = false;

	while( ! vidcap->eof() )
	{   
		// capture a video frame
		if( mouse_faceparam.play == true)
			pImg = vidcap->query();
		else 
			continue;

		if ( pImg == NULL )
			continue;

		// make a copy, flip if upside-down
		CvImage image( cvGetSize(pImg), pImg->depth, pImg->nChannels );
		if( pImg->origin == IPL_ORIGIN_BL ) //flip live camera's frame
			cvFlip( pImg, image );
		else
			cvCopy( pImg, image );

		// convert to gray_image for face analysis
		CvImage gray_image( image.size(), image.depth(), 1 );
		if( image.channels() == 3 )
			cvCvtColor( image, gray_image, CV_BGR2GRAY );
		else
			cvCopy( image, gray_image );

		// do face tracking
		start_ticks = ticks = cvGetTickCount();	
       
		int face_num = 0;
        if( is_piclist )
            face_num = detector.detect( gray_image, rects, face_max );
        else
            face_num = tracker.track( gray_image, rects, face_max, image ); // track in a video for faster speed
		  //face_num = tracker.detect( gray_image, rects, face_max ); // detect in an image

		//set param for mouse event processing
		if(!quiet)
		{
			mouse_faceparam.face_num = face_num;
			mouse_faceparam.image    = image;
		}

		ticks       = cvGetTickCount() - ticks;
		tracker_fps = 1000.0 / ( 1e-3 * ticks / cvGetTickFrequency() );
		tracker_total_ticks += ticks;

        if( fp_imaginfo != NULL )
            fprintf( fp_imaginfo, "%s  %d", vidcap->filename(), face_num );

        // blink/smile/gender/age/face recognize section
		for( int i=0; i<face_num; i++ )
		//for( int i=0; i< MIN(1,face_num); i++ )
		{
			// get face rect and id from face tracker
			CvRect rect = rects[i].rc;

            if( fp_imaginfo != NULL )
                fprintf( fp_imaginfo, "  %d %d %d %d %f", rect.x, rect.y, rect.width, rect.height, rects[i].prob );

			int    face_trackid = rects[i].fid;
			float  like = rects[i].prob;
			int    angle= rects[i].angle;

			// filter out outer faces
			if( rect.x+rect.width  > gray_image.width()   || rect.x < 0 ) continue;
			if( rect.y+rect.height > gray_image.height() || rect.y < 0 ) continue;

			//tracker.getThumbnail(image, rect, thumbnailImg);

			// detect landmark points 
			ticks = cvGetTickCount();	

			if(bRunLandmark)
			{
                if( is_piclist )
				    bLandmark = landmarkDetector.detect( gray_image, &rect, landmark6, parameters, angle ); //detect in an image
                else
				    bLandmark = landmarkDetector.track( gray_image, &rect, landmark6, parameters, angle ); // track in a video for faster speed

				ticks = cvGetTickCount() - ticks;
				landmark_fps = 1000.0 / ( 1e-3 * ticks / cvGetTickFrequency() );
				landmark_total_ticks += ticks;
			}
			else
				bLandmark = false;

	
			if(quiet == false && bLandmark == false) 
			{
				//DrawFaceRect
				cxlibDrawFaceRect(image, rect);
				continue;
			}

			// warped align face and hist eq to delighting
			ticks = cvGetTickCount();	

			cutFace.init(gray_image, rect, landmark6);

			ticks = cvGetTickCount() - ticks;
			if(ticks > 1)
				align_fps = 1000.0 / ( 1e-3 * ticks / cvGetTickFrequency() );
			else
			{	align_fps = 0;
				ticks = 0;
			}
			align_total_ticks += ticks;

			if(saveface)   //save face icon for training later
			{
				//save cutfaces
				if(sfolder)
				{
#ifdef WIN32
					_splitpath(vidcap->filename(),driver,dir,fname,ext);
					sprintf(sImgPath, "%s//%s%s", sfolder, fname,ext);
#else
					sprintf(sImgPath, "%s//%06d.jpg", sfolder, faceimgID++);
#endif
				}
				else
					sprintf(sImgPath, "%s#.jpg", vidcap->filename());
				
				cvSaveImage(sImgPath, cutFace.getBigCutFace());
			}

			// detect blink
			bBlink = 0;	
			probBlink = 0;
			if(blink && bLandmark)
			{
				ticks = cvGetTickCount();	
				float blink_threshold = blinkDetector.getDefThreshold();//0.5;
				int ret = blinkDetector.predict( &cutFace, &probBlink);
			
				if(probBlink > blink_threshold )
					bBlink = 1;  //eye close
				else 
					bBlink = -1; //eye open

				ticks = cvGetTickCount() - ticks;
				blink_fps = 1000.0/(1e-3*ticks/cvGetTickFrequency());
				blink_total_ticks += ticks;

				print_score = probBlink;
			}
			else blink_fps = 0;

			// detect smile
			bSmile    = 0;	
			probSmile = 0;
			if ( smile && bLandmark )
			{	
				ticks = cvGetTickCount();
				float smile_threshold = smileDetector.getDefThreshold(); //0.48;  
				int ret = smileDetector.predict(&cutFace, &probSmile);

				if(probSmile > smile_threshold)
					bSmile = 1;  //smile
				else 
					bSmile = -1; //not smile

				ticks	  = cvGetTickCount() - ticks;
				smile_fps = 1000.0 /( 1e-3 * ticks / cvGetTickFrequency() );
				smile_total_ticks += ticks;

				print_score = probSmile;
			}
			else smile_fps = 0;

			//detect gender
			bGender    = 0;	
			probGender = 0;
			if(gender && bLandmark)
			{
				ticks = cvGetTickCount();	
				float gender_threshold = genderDetector.getDefThreshold(); // 0.42; 
				int ret = genderDetector.predict(&cutFace, &probGender);

				if(probGender > gender_threshold)
					bGender =  1; //female
				else
					bGender = -1; //male

				//bGender = -1:male, 1:female, 0: null
				// smooth prediction result
                if( ! is_piclist )
				    bGender = genderDetector.voteLabel(face_trackid, bGender);
				
				ticks = cvGetTickCount() - ticks;
				gender_fps = 1000.0/(1e-3*ticks/cvGetTickFrequency());
				gender_total_ticks += ticks;

				print_score = probGender; 
			}
			else gender_fps = 0;

			//detect age
			nAgeID  = -1;
			if(age && bLandmark && rect.width*rect.height > 40*40)
			{
				ticks = cvGetTickCount();	

				//nAgeID = 0:"Baby", 1:"Kid", 2:"Adult", 3:"Senior"
				nAgeID = ageDetector.predict(&cutFace, probAge);

				// smooth prediction result
                if( ! is_piclist )
				    nAgeID = ageDetector.voteLabel(face_trackid, nAgeID); 

				ticks = cvGetTickCount() - ticks;
				age_fps = 1000.0/(1e-3*ticks/cvGetTickFrequency());
				age_total_ticks += ticks;

				print_score = probAge[nAgeID]; 
				//if( ! quiet )	cxDrawAignFace2Image(image, pCutFace2);
			}
			else 
			{
				age_fps = 0;
			}

			// recognize the face id
			// we only do recognition every 5 frames,interval
			char  *sFaceCaption = NULL;
			char  sFaceCaptionBuff[256];
            int face_id = 0;
			probFaceID = 0;
			if ( recog && bLandmark )
			{
				ticks = cvGetTickCount();
				float face_threshold = faceRecognizer.getDefThreshold(); 
				/////////////////////////////////////////////////////////////////////////////////////////
				int face_id  = -1;
				if(bEnableAutoCluster & !is_piclist)
				{
					bool bAutocluster = true;
					if(mouse_faceparam.ret_online_collecting) bAutocluster = false;
					//face clustering
					face_id  = faceRecognizer.predict(&cutFace, &probFaceID, bAutocluster, face_trackid, frames);
				}
				else//face recognition
					face_id  = faceRecognizer.predict(&cutFace, &probFaceID);
				/////////////////////////////////////////////////////////////////////////////////////////

				ticks    = cvGetTickCount() - ticks;
				recg_fps = 1000.0f / ( 1e-3 * ticks / cvGetTickFrequency() );
				recg_total_ticks += ticks;
				
				// smooth prediction result
                if( ! is_piclist && !bEnableAutoCluster)
                {
				    if(probFaceID > face_threshold*1.0)
					    face_id = faceRecognizer.voteLabel(face_trackid, face_id); 
				    else
					    face_id = faceRecognizer.voteLabel(face_trackid, -1);
                }
				else if(probFaceID <= face_threshold)
				{
					face_id =-1;
				}

				//set face name caption
				if(face_id >= 0)
				{
					// recognized face name
					const char* sFaceName = faceRecognizer.getFaceName(face_id);
					sprintf(sFaceCaptionBuff, "%s %.2f", sFaceName, probFaceID);
					//sprintf(sFaceCaptionBuff, "%s", sFaceName); //dispaly score
					sFaceCaption = sFaceCaptionBuff;
					
					print_score  = probFaceID;
					print_faceid = face_id;
				}
				else
				{   // failed to recognize 
					//sprintf(sFaceCaptionBuff, "N\A %.2f", probFaceID);
					//sFaceCaption = sFaceCaptionBuff;
				}

				// collect and save unknown face exemplars
				if(probFaceID < face_threshold*0.9 || face_id != mouse_faceparam.ret_faceset_id )
				{
					if(mouse_faceparam.ret_online_collecting && (face_num ==1 || face_trackid == mouse_faceparam.ret_facetrack_id))
					{
						if( rect.x > gray_image.width()/4 && rect.x+rect.width < gray_image.width()*3/4 ) 
						{
							mouse_faceparam.updated = true;
							int nFaceSetIdx = faceRecognizer.getFaceSetIdx(mouse_faceparam.ret_faceset_id);
							bool bflag = faceRecognizer.tryInsertFace(cutFace.getBigCutFace(), nFaceSetIdx);
							//printf("insert flag %d\n", bflag);
						}
					}
				}
			}
			else recg_fps = 0;

			if( ! quiet )
			{
				sprintf(sCaptionInfo, "FPS: %03d Fd:%04d Ld:%04d Fa:%04d Bl:%04d Sm:%04d Ge:%04d Ag:%03d Rc:%03d",
					(int)frame_fps, (int)tracker_fps, (int)landmark_fps, (int)align_fps, 
					(int)blink_fps,   (int)smile_fps,    (int)gender_fps, (int)age_fps, (int)recg_fps);

				//sprintf(sFaceCaptionBuff, "%.2f", print_score);
				//sFaceCaption = sFaceCaptionBuff;

				int trackid = -1; //face_trackid. don't display trackid if -1
				cxlibDrawFaceBlob( image, pFont, trackid, rect, landmark6, probSmile, 
					bBlink, bSmile, bGender, nAgeID, sFaceCaption, NULL,
					pImgSmileBGR, pImgSmileBGRA, pImgSmileMask);
			}

            // log file
            if( fp_faceinfo != NULL )
            {
                // index,  rect,  landmark6,  bBlink, probBlink, bSmile, probSmile, bGender, probGender, nAgeID, probAge[nAgeID], face_id, probFaceID
				//fprintf( fp_faceinfo, "#%s# @%s@ ",    vidcap->filename(), sImgPath);
				fprintf( fp_faceinfo, "#%s# ",    vidcap->filename());
                fprintf( fp_faceinfo, "faceidx=( %06d %02d )", vidcap->index(), i+1 );
				fprintf( fp_faceinfo, "   rect=( %3d %3d %3d %3d )", rect.x, rect.y, rect.width, rect.height );
                fprintf( fp_faceinfo, "   landmark6=(" );
                int l;
                for( l = 0; l < 6; l++ )
                    fprintf( fp_faceinfo, " %3.0f %3.0f", landmark6[l].x, landmark6[l].y );
                fprintf( fp_faceinfo, " )");
                fprintf( fp_faceinfo, "   blink=( %+d %f )", bBlink, probBlink );
                fprintf( fp_faceinfo, "   smile=( %+d %f )", bSmile, probSmile );
                fprintf( fp_faceinfo, "   gender=( %+d %f )", bGender, probGender );
                fprintf( fp_faceinfo, "   agegroup=( %+d %f )", nAgeID, (nAgeID >= 0 && nAgeID < 4) ? probAge[nAgeID] : 1.0f );
                fprintf( fp_faceinfo, "   identity=( %+d %f )", face_id, probFaceID );
                fprintf( fp_faceinfo, "\n" );
            }
        }
        if( fp_imaginfo != NULL )
            fprintf( fp_imaginfo, "\n" );

		ticks    = cvGetTickCount() - start_ticks;
		total_ticks += (ticks);
		frame_fps = 1000.0f / ( 1e-3 * ticks / cvGetTickFrequency() );

		// frame face_num
		frames++;

		//auto focus faces
		if(quiet == false && bAutoFocus)
		{
			if(imgAutoFocus)
				cvCopy(image, imgAutoFocus);
			else
				imgAutoFocus = cvCloneImage(image);
			cxlibAutoFocusFaceImage(imgAutoFocus, image, rects, face_num);
		}

		// next frame if quiet
		if( quiet )
			continue;
		else
		{
			// draw status info for custom interaction
			if(mouse_faceparam.ret_online_collecting == 1)
			{
				sprintf(sCaptionInfo, "Collecting faces for track_id = %d", mouse_faceparam.ret_facetrack_id);
				//draw face collecting region
				cvLine(image, cvPoint(image.width()/4, 0), cvPoint(image.width()/4, image.height()-1), CV_RGB(255,255,0), 2);
				cvLine(image, cvPoint(image.width()*3/4, 0), cvPoint(image.width()*3/4, image.height()-1), CV_RGB(255,255,0), 2);
			}
			else
				sprintf(sCaptionInfo, "FPS: %03d Fd:%04d Ld:%04d Fa:%04d Bl:%04d Sm:%04d Ge:%04d Ag:%03d Rc:%03d",
					(int)frame_fps, (int)tracker_fps, (int)landmark_fps, (int)align_fps, 
					(int)blink_fps,   (int)smile_fps,    (int)gender_fps, (int)age_fps, (int)recg_fps);

			cxlibDrawCaption( image, pFont, sCaptionInfo);
		}
	
		//show Image
		if (image.width() <= 800)
		{
			//show image
			cvShowImage( str_title, image );
		}
		else
		{   // show scaled smaller image
			CvImage scale_image (cvSize(800, image.height()*800/image.width()), image.depth(), 3 );
			cvResize (image, scale_image);
			cvShowImage( str_title, scale_image );
		}

		// user interaction
		int key = cvWaitKey( 30 );
		//int key = cvWaitKey( );
		if( key == ' ' ) // press the spacebar to pause the video play 
			cvWaitKey( 0 );                           
		else if( key == 27 )
			break;	    // press 'esc' to exit
		else if( key == 'a' )
		{  // add new face name
			if(face_num > 0)
			{   
				CvRect rect = rects[0].rc;
				int x = rect.x+rect.width/2;
				int y = rect.y+rect.height/2;
				addFaceSet( x, y, &mouse_faceparam);
			}
		}
		else if( key == 'c' )
		{   // collect face exemplars for current selected facename
			mouse_faceparam.ret_online_collecting = 1; //enable online face exemplar collecting
		}
		else if( key == 'z' )
			// turn on/off the autofocus flag
			bAutoFocus = !bAutoFocus;
		else if(key >= 0)
		{
			if(mouse_faceparam.ret_online_collecting == 1)
			{   // stop collecting faces
				mouse_faceparam.ret_online_collecting = 0; //disable online face exemplar collecting
				mouse_faceparam.ret_facetrack_id = -1;
			}

			if( key == 's')
			{   // save face models
				faceRecognizer.saveFaceModelXML("faceset_model.xml");
				sprintf(sCaptionInfo, "%s", "saved the face model");
				cxlibDrawCaption( pImg, pFont, sCaptionInfo);
				cvShowImage( str_title, pImg );
				cvWaitKey( 400 ); 
			}
		}
	}

	// print speed info about fps
	float temp    = 1e-6f / cvGetTickFrequency();
	tracker_fps   = 1.0f  / ( tracker_total_ticks * temp / frames );

	if (landmark_total_ticks != 0.0)
		landmark_fps = 1.0f  / ( landmark_total_ticks * temp / frames );

	if (align_total_ticks != 0.0)
		align_fps    = 1.0f  / ( align_total_ticks * temp / frames );

	if (blink_total_ticks != 0.0)
		blink_fps  = 1.0f  / (blink_total_ticks * temp / frames);

	if (smile_total_ticks != 0.0)
		smile_fps  = 1.0f  / (smile_total_ticks * temp / frames);

	if (gender_total_ticks != 0.0)
		gender_fps = 1.0f  / (gender_total_ticks * temp / frames);

	if (age_total_ticks != 0.0)
		age_fps = 1.0f  / (age_total_ticks * temp / frames);

	if (recg_total_ticks != 0.0)
		recg_fps   = 1.0f  / (recg_total_ticks  * temp / frames);

	total_fps = 1.0f / (total_ticks * temp / frames);

	printf( "Total frames:%d  Speed:%.1f fps\n", frames, total_fps);
	printf( "FPS: Fd:%.1f Ld:%.1f Fa:%.1f Bl:%.1f Sm:%.1f Ge:%.1f Ag:%.1f Rc:%.1f",
		tracker_fps, landmark_fps, align_fps, 
		blink_fps,   smile_fps,    gender_fps, age_fps, recg_fps);

	//save updated face model
	if(mouse_faceparam.updated == true)
	{
		sprintf(sCaptionInfo, "%s", "press key 's' to save updated face model or other keys to cancel");
		cxlibDrawCaption( pImg, pFont, sCaptionInfo);
		cvShowImage( str_title, pImg );

		int key = cvWaitKey();
		if( key == 's')
			faceRecognizer.saveFaceModelXML("faceset_model.xml");
	}

	
	//save merged face model for dynamic clustering of smoothID
	vFaceSet vMergedFaceSet;
	int minWeight = 10;
	faceRecognizer.getMergedFaceSet(vMergedFaceSet, minWeight);
	faceRecognizer.saveFaceModelXML("faceset_modelMerged.xml", &vMergedFaceSet);
	//faceRecognizer.saveFaceModelXML("faceset_modelMerged#.xml");

	//release buff 
	
	//release global GUI data
	if( !quiet )
		cvDestroyWindow( str_title );

	cvReleaseImage(&thumbnailImg);
	cvReleaseImage(&pImgSmileBGR);
	cvReleaseImage(&pImgSmileBGRA);
	cvReleaseImage(&pImgSmileMask);
	
	delete pFont;

    delete vidcap;

    if( fp_imaginfo != NULL )
        fclose( fp_imaginfo );
	
    if( fp_faceinfo != NULL )
        fclose( fp_faceinfo );

    return 0;
}

int testfaceLib_pThread ( const char* str_video, int trackerType, int multiviewType, int recognizerType, const char* str_facesetxml, int threads, 
						 bool blink, bool smile, bool gender, bool age, bool recog, bool quiet, bool saveface, const char* sfolder, bool bEnableAutoCluster )
{
    FILE* fp_imaginfo = fopen( "imaginfo.txt", "w" );

	bool bAutoFocus = false;
	IplImage *imgAutoFocus = NULL;

	int  sampleRate =1;
	
	if(str_facesetxml == NULL)
		str_facesetxml = "faceset_model.xml";

	int  prob_estimate[7];
	char sState[256];
	EnumViewAngle  viewAngle = (EnumViewAngle)multiviewType;
	//dynamic clustering for smooth ID registration
	//bEnableAutoCluster =  true;

	CxlibFaceAnalyzer faceAnalyzer(viewAngle, (EnumTrackerType)trackerType, blink, smile, gender, age, recog, sampleRate, str_facesetxml, recognizerType, bEnableAutoCluster); 

	/////////////////////////////////////////////////////////////////////////////////////
	//	init GUI window
	const char* str_title = "Face Tester";
	if( ! quiet )
		cvNamedWindow( str_title, CV_WINDOW_AUTOSIZE );

	char sCaptionInfo[256] = "";
	CvFont *pFont = new CvFont;
	cvInitFont(pFont, CV_FONT_HERSHEY_PLAIN, 0.85, 0.85, 0, 1);

	// load GUI smile icon images
	IplImage *pImgSmileBGR;
	IplImage *pImgSmileMask;
	if(age == 0)
	{   // smile icon
		pImgSmileBGR  = cvLoadImage( "smile.bmp" );
		pImgSmileMask = cvLoadImage( "smilemask.bmp", 0 );
	}
	else
	{   // gender/age/smile icons
		pImgSmileBGR  = cvLoadImage( "faceicon.bmp" );
		pImgSmileMask = cvLoadImage( "faceiconMask.bmp", 0 );
	}

	IplImage *pImgSmileBGRA = cvCreateImage( cvSize(pImgSmileBGR->width, pImgSmileBGR->height), IPL_DEPTH_8U, 4 );
	cvCvtColor(pImgSmileBGR, pImgSmileBGRA, CV_BGR2BGRA );

	// open video source
    size_t len = strlen( str_video );
    bool is_piclist = (0 == stricmp( str_video + len - 4, ".txt" ));
    CxImageSeqReader* vidcap = NULL;
    if( is_piclist )
        vidcap = new CxPicListReader( str_video );
    else
        vidcap = new CxVideoReader( str_video );
	if( cvGetErrStatus() < 0 )
	{   
		cvSetErrStatus( CV_StsOk );
		return -1;
	}

	// when using camera, set to 640x480, 30fps
	if( isdigit(str_video[0]) != 0 && str_video[1] == '\0' )
	{
		vidcap->width( 640 );
		vidcap->height( 480 );
		vidcap->fps( 30 );
	}

	// print beginning info
	printf( "tracker cascade:  '%s'\n", trackerType== TRA_HAAR ? "haar" : (recognizerType== TRA_SURF ? "surf" : "pf tracker SURF"));
	printf( "face recognizer:  '%s'\n", recognizerType == RECOGNIZER_BOOST_GB240 ? "boost gabor240" : "cascade gloh"  );
	printf( "video:    '%s', %dx%d, %2.1f fps\n", str_video, 
		vidcap->width(), vidcap->height(), vidcap->fps() );

	// set mouse event process
	CxMouseParam mouse_faceparam;
	mouse_faceparam.updated = false;
	mouse_faceparam.play    = true;
	mouse_faceparam.ret_online_collecting = 0;

	static const int MAX_FACES = 16; 
	if(! quiet)
	{
		mouse_faceparam.play    = true;
		mouse_faceparam.updated = false;
		mouse_faceparam.face_num  = faceAnalyzer.getMaxFaceNum();
		mouse_faceparam.rects     = faceAnalyzer.getFaceRects();
		mouse_faceparam.image     = NULL;
		mouse_faceparam.cut_big_face= faceAnalyzer.getBigCutFace();
		mouse_faceparam.typeRecognizer = 0;
		mouse_faceparam.faceRecognizer = &faceAnalyzer;
		mouse_faceparam.ret_online_collecting = 0;
		cvSetMouseCallback(	str_title, my_mouse_callback, (void*)&mouse_faceparam );
		faceAnalyzer.setMouseParam(&mouse_faceparam);
	}

	// init count ticks                   
	int64  ticks, start_ticks, total_ticks;
	int64  tracker_total_ticks;
	double tracker_fps, total_fps; 

	start_ticks         = total_ticks  = 0;
	tracker_total_ticks = 0;
		
	// loop for each frame of a video/camera
	int frames = 0;
	IplImage *pImg = NULL;

	while( ! vidcap->eof() )
	{   
		// capture a video frame
		if( mouse_faceparam.play == true)
			pImg = vidcap->query();
		else 
			continue;

		if ( pImg == NULL )
			break;

		// make a copy, flip if upside-down
		CvImage image( cvGetSize(pImg), pImg->depth, pImg->nChannels );
		if( pImg->origin == IPL_ORIGIN_BL ) //flip live camera's frame
			cvFlip( pImg, image );
		else
			cvCopy( pImg, image );

		// convert to gray_image for face analysis
		CvImage gray_image( image.size(), image.depth(), 1 );
		if( image.channels() == 3 )
			cvCvtColor( image, gray_image, CV_BGR2GRAY );
		else
			cvCopy( image, gray_image );

		///////////////////////////////////////////////////////////////////
		// do face tracking and face recognition
		start_ticks = ticks = cvGetTickCount();	

        if( is_piclist )
            faceAnalyzer.detect(gray_image, prob_estimate, sState);
        else
		    faceAnalyzer.track(gray_image, prob_estimate, sState, image);   // track face in each frame but recognize by pthread
		//faceAnalyzer.detect(gray_image, prob_estimate, sState);// track and recognizer face in each frame 

		int face_num = faceAnalyzer.getFaceNum();

		ticks       = cvGetTickCount() - ticks;
		tracker_fps = 1000.0 / ( 1e-3 * ticks / cvGetTickFrequency() );
		tracker_total_ticks += ticks;

		
		//set param for mouse event processing
		if(!quiet)
		{
			mouse_faceparam.face_num = face_num;
			mouse_faceparam.image    = image;
		}

        if( fp_imaginfo != NULL )
            fprintf( fp_imaginfo, "%s  %d", vidcap->filename(), face_num );

		// blink/smile/gender/age/face recognize section
		for( int i=0; i<face_num; i++ )
		{
			// get face rect and id from face tracker
			CvRectItem rectItem = faceAnalyzer.getFaceRect(i);
			CvRect rect = rectItem.rc;
			int    face_trackid = rectItem.fid;
			float  probSmile = faceAnalyzer.getFaceSmileProb(i);
			int    bBlink  = faceAnalyzer.getFaceBlink(i);
			int    bSmile  = faceAnalyzer.getFaceSmile(i);
			int    bGender = faceAnalyzer.getFaceGender(i);
			int    nAgeID  = faceAnalyzer.getFaceAge(i);
			int    nFaceID = faceAnalyzer.getFaceID(i);
			float  fFaceProb= faceAnalyzer.getFaceProb(i);
			
			char *sFaceCaption = NULL;
			char sFaceNameBuff[256];
			char *sFaceName = faceAnalyzer.getFaceName(i);
			if(sFaceName[0] != '\0')
			{
				sprintf(sFaceNameBuff, "%s %.2f", sFaceName, fFaceProb);
				sFaceCaption = sFaceName;
				sFaceCaption = sFaceNameBuff;
			}

			if( ! quiet )
			{
				CvPoint2D32f *landmark6 = NULL;
				sprintf(sCaptionInfo, "FPS:%04d, %s", (int)tracker_fps, sState);

				int trackid = -1; //face_trackid , don't display trackid if -1
				cxlibDrawFaceBlob( image, pFont, trackid, rect, landmark6, probSmile, 
					bBlink, bSmile, bGender, nAgeID, sFaceCaption, NULL,
					pImgSmileBGR, pImgSmileBGRA, pImgSmileMask);
			}

            if( fp_imaginfo != NULL )
                fprintf( fp_imaginfo, "  %d %d %d %d", rect.x, rect.y, rect.width, rect.height );
		}
        if( fp_imaginfo != NULL )
            fprintf( fp_imaginfo, "\n" );

		///////////////////////////////////////////////////////////////////
		total_ticks += (cvGetTickCount() - start_ticks);
		
		// frame face_num
		frames++;

		//auto focus faces
		if(quiet == false && bAutoFocus)
		{
			if(imgAutoFocus)
				cvCopy(image, imgAutoFocus);
			else
				imgAutoFocus = cvCloneImage(image);

			CvRectItem *rects = faceAnalyzer.getFaceRects();
			cxlibAutoFocusFaceImage(imgAutoFocus, image, rects, face_num);
		}

		// next frame if quiet
		if( quiet )
			continue;
		else
		{
			// draw status info for custom interaction
			if(mouse_faceparam.ret_online_collecting == 1)
			{
				sprintf(sCaptionInfo, "Collecting faces for track_id = %d", mouse_faceparam.ret_facetrack_id);
				//draw face collecting region
				cvLine(image, cvPoint(image.width()/4, 0), cvPoint(image.width()/4, image.height()-1), CV_RGB(255,255,0), 2);
				cvLine(image, cvPoint(image.width()*3/4, 0), cvPoint(image.width()*3/4, image.height()-1), CV_RGB(255,255,0), 2);
			}
			else
				sprintf(sCaptionInfo, "FPS:%04d, %s", (int)tracker_fps, sState);

			cxlibDrawCaption( image, pFont, sCaptionInfo);
		}
		
		//show Image
		if (image.width() <= 800)
			cvShowImage( str_title, image );
		else
		{   // display scaled smaller aimge
			CvImage scale_image (cvSize(800, image.height()*800/image.width()), image.depth(), 3 );
			cvResize (image, scale_image);
			cvShowImage( str_title, scale_image );
		}

		// user interaction
		int key = cvWaitKey(1);
		//int key = cvWaitKey(0);
		if( key == ' ' )     // press space bar to pause the video play
			cvWaitKey( 0 );                           
		else if( key == 27 ) // press 'esc' to exit
			break;	                                   
		else if( key == 'a' )
		{  // add new face name
			if(face_num > 0)
			{   
				CvRect rect = faceAnalyzer.getFaceRect(0).rc;
				int x = rect.x+rect.width/2;
				int y = rect.y+rect.height/2;
				addFaceSet( x, y, &mouse_faceparam);
			}
		}
		else if( key == 'c' )
		{   //enable flag to collect face exemplars for the selected face name
			mouse_faceparam.ret_online_collecting = 1; //enable online face exemplar collecting
		}
		else if( key == 'z' )
			bAutoFocus = !bAutoFocus;
		else if(key >= 0)
		{
			if(mouse_faceparam.ret_online_collecting == 1)
			{   // stop collecting face exemplars
				mouse_faceparam.ret_online_collecting = 0; //disable online face exemplar collecting
				mouse_faceparam.ret_facetrack_id = -1;
			}

			if( key == 's')
			{
				// save faceset xml model
				faceAnalyzer.saveFaceModelXML("faceset_model.xml");
				sprintf(sCaptionInfo, "%s", "saved the face model");
				cxlibDrawCaption( pImg, pFont, sCaptionInfo);
				cvShowImage( str_title, pImg );
				cvWaitKey( 400 ); 
			}
		}
	}

	// print info about fps
	float temp    = 1e-6f / cvGetTickFrequency();
	tracker_fps   = 1.0f  / ( tracker_total_ticks * temp / frames );
	
	total_fps = 1.0f / (total_ticks * temp / frames);

	printf( "Total frames:%d  Speed:%.1f fps\n", frames, total_fps);
	printf( "FPS: %.1f ", tracker_fps);

	//save updated faceset model
	if(mouse_faceparam.updated == true)
	{
		sprintf(sCaptionInfo, "%s", "press key 's' to save updated face model or other keys to cancel");
		cxlibDrawCaption( pImg, pFont, sCaptionInfo);
		cvShowImage( str_title, pImg );

		int key = cvWaitKey();
		if( key == 's')
			faceAnalyzer.saveFaceModelXML("faceset_model.xml");
	}

	//save merged face model for dynamic clustering of smoothID
	vFaceSet vMergedFaceSet;
	int minWeight =10; 
	faceAnalyzer.getMergedFaceSet(vMergedFaceSet, minWeight);
	faceAnalyzer.saveFaceModelXML("faceset_modelMerged.xml", &vMergedFaceSet);

	//release global GUI data
	if( !quiet )
		cvDestroyWindow( str_title );

	cvReleaseImage(&pImgSmileBGR);
	cvReleaseImage(&pImgSmileBGRA);
	cvReleaseImage(&pImgSmileMask);
	delete pFont;

    delete vidcap;

    if( fp_imaginfo != NULL )
        fclose( fp_imaginfo );

    return 0;
}

// main test example of facelib
int consoleTestFaceLib( const char* str_video, int  trackerType, int multiviewType, int recognizerType, const char* str_facesetxml, int threads, 
				bool blink, bool smile, bool gender, bool age, bool recog, bool quiet, bool saveface, const char* sfolder, bool bEnableAutoCluster )

{
	if(threads == 1) 
		// detect and recognize each video frame
		testfaceLib_sThread(str_video, trackerType, multiviewType, recognizerType, str_facesetxml, threads, 
			blink, smile, gender, age, recog, quiet, saveface, sfolder, bEnableAutoCluster );
	else  
		//detect each video frame and use another pthread to detect landmark/blink/smile/gender/age and recognize faces
		testfaceLib_pThread(str_video, trackerType, multiviewType, recognizerType, str_facesetxml, threads, 
			blink, smile, gender, age, recog, quiet, saveface, sfolder, bEnableAutoCluster );
	
	return 0;
}


