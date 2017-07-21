
/*
 * Ubitrack - Library for Ubiquitous Tracking
 * Copyright 2006, Technische Universitaet Muenchen, and individual
 * contributors as indicated by the @authors tag. See the
 * copyright.txt in the distribution for a full listing of individual
 * contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */


/**
 * @ingroup dataflow_components
 * @file
 * StereoMatching on GPU.
 *
 * @author Ulrich Eck <ueck@net-labs.de>
 */

#include <string>
#include <list>
#include <iostream>

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>
#include <log4cpp/Category.hh>

#include <utDataflow/PushSupplier.h>
#include <utDataflow/PushConsumer.h>
#include <utDataflow/TriggerComponent.h>
#include <utDataflow/TriggerInPort.h>
#include <utDataflow/TriggerOutPort.h>
#include <utDataflow/ComponentFactory.h>
#include <utMeasurement/Measurement.h>
#include <utUtil/OS.h>
#include <utUtil/Exception.h>
#include <utVision/Image.h>


#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stereo.hpp"

#include <stdio.h>


using namespace Ubitrack;
using namespace Ubitrack::Vision;
using namespace Ubitrack::Measurement;
using namespace cv;


// get a logger
// log4cpp
#include <log4cpp/Category.hh>
static log4cpp::Category& logger( log4cpp::Category::getInstance( "Ubitrack.StereoVision.StereoMatchGPU" ) );

namespace Ubitrack { namespace StereoVision {

/**
 * @ingroup dataflow_components
 * StereoMatching on GPU.
 * This class computes a disparity map using stereo-vision cameras with the OpenCV GPU-based algorithms
 *
 * @par Input Ports
 * ExpansionInPort<Image> with name "ImageLeft".
 * ExpansionInPort<Image> with name "ImageRight".
 *
 * To be completed
 *
 * @par Operation
 * TBD
 */

class StereoMatchGPU
    : public Dataflow::TriggerComponent
{

protected:

    /** still uses deprecated camera intrinsics data .. */

    /** intrinsic camera matrix */
    Dataflow::PullConsumer< Measurement::CameraIntrinsics > m_inPortIntrinsicsLeft;
    Dataflow::PullConsumer< Measurement::CameraIntrinsics > m_inPortIntrinsicsRight;

    /** left-to-right camera Transform */
    Dataflow::PullConsumer< Measurement::Pose > m_inPortCameraOffset;

    /** Input port camera left of the component. */
    Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_inPortImageLeft;

    /** Input port camera right of the component. */
    Dataflow::TriggerInPort< Measurement::ImageMeasurement > m_inPortImageRight;

    /** Output port of the component for the disparity Image . */
    Dataflow::TriggerOutPort< Measurement::ImageMeasurement > m_outPortDisparity;

    /** Output port of the component for the 4x4 Transform matrix, that is required to generate 3d points from the disparity image*/
    Dataflow::PushSupplier< Measurement::Matrix4x4 > m_outPortQMatrix;

    /** Output port of the component for the undistorted and rectified camera image left*/
    Dataflow::PushSupplier< Measurement::ImageMeasurement > m_outPortImageLeft;

    /** Output port of the component for the undistorted and rectified camera image right*/
    Dataflow::PushSupplier< Measurement::ImageMeasurement > m_outPortImageRight;

    // already initialized?
    bool m_initialized;

	// need better names .. just for testing now ..
    // Mat map11, map12, map21, map22;
    // cuda::GpuMat d_left, d_right;
    // cuda::GpuMat d_disp;

    // cuda::StereoBM bm;
    // cuda::StereoBeliefPropagation bp;
    // cuda::StereoConstantSpaceBP csbp;

 
    int m_ndisparity;
    enum {BM, BP, CSBP} m_method;

    int m_bmFilter;
    int m_windowSize;
    int m_iterations;
    int m_levels;
    

public:
    /**
     * UTQL component constructor.
     *
     * @param sName Unique name of the component.
     * @param subgraph UTQL subgraph
     */
    StereoMatchGPU( const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > subgraph )
        : Dataflow::TriggerComponent( sName, subgraph )
        , m_inPortIntrinsicsLeft( "IntrinsicsLeft", *this )
        , m_inPortIntrinsicsRight( "IntrinsicsRight", *this )
        , m_inPortCameraOffset( "CameraOffset", *this )
        , m_inPortImageLeft( "ImageLeftIn", *this )
        , m_inPortImageRight( "ImageRightIn", *this )
        , m_outPortDisparity( "DisparityImage", *this )
        , m_outPortQMatrix( "QMatrix", *this )
        , m_outPortImageLeft( "ImageLeftOut", *this )
        , m_outPortImageRight( "ImageRightOut", *this )
        , m_initialized( false )
        , m_ndisparity( 64 )
        , m_method( BP )
	, m_bmFilter( StereoBM::PREFILTER_NORMALIZED_RESPONSE )
	, m_windowSize( 19 )
	, m_iterations( 5 )
 	, m_levels( 5 )
    {

        if( subgraph->m_DataflowAttributes.hasAttribute( "nDisparity" ) )
            subgraph->m_DataflowAttributes.getAttributeData( "nDisparity", m_ndisparity );

        if( subgraph->m_DataflowAttributes.hasAttribute( "algorithm" ) ) {
            std::string algorithm = subgraph->m_DataflowAttributes.getAttributeString( "algorithm" );
        if (algorithm == "bm") {
    		m_method = BM;
    		if ( subgraph->m_DataflowAttributes.hasAttribute( "windowSize" ) ) {
    			subgraph->m_DataflowAttributes.getAttributeData( "windowSize", m_windowSize );
    		}
            if ( subgraph->m_DataflowAttributes.hasAttribute( "bmFilter" ) ) {
    			if ( subgraph->m_DataflowAttributes.getAttributeString( "bmFilter" ) == "xsobel" ) {
    				m_bmFilter = StereoBM::PREFILTER_XSOBEL;
    			}
    		}

        } else if (algorithm == "bp") {
        m_method = BP;
                if ( subgraph->m_DataflowAttributes.hasAttribute( "iterations" ) ) {
                        subgraph->m_DataflowAttributes.getAttributeData( "iterations", m_iterations );
                }
                if ( subgraph->m_DataflowAttributes.hasAttribute( "levels" ) ) {
                        subgraph->m_DataflowAttributes.getAttributeData( "levels", m_levels );
                }

	    } else if (algorithm == "csbp") {
		m_method = CSBP;
                if ( subgraph->m_DataflowAttributes.hasAttribute( "iterations" ) ) {
                        subgraph->m_DataflowAttributes.getAttributeData( "iterations", m_iterations );
                }
                if ( subgraph->m_DataflowAttributes.hasAttribute( "levels" ) ) {
                        subgraph->m_DataflowAttributes.getAttributeData( "levels", m_levels );
                }

	    }
	}

    }

    /** Method that computes the result. */
    void compute( Measurement::Timestamp ts )
    {

        boost::shared_ptr< Vision::Image > vimgLeft = ( m_inPortImageLeft.get()->Clone() );
        boost::shared_ptr< Vision::Image > vimgRight = ( m_inPortImageRight.get()->Clone() );

		cv::Mat imgLeft( vimgLeft->Mat(), false );
        cv::Mat imgRight( vimgRight->Mat(), false );

        if (!m_initialized) {

            Mat M1, D1, M2, D2, R, T;
            Size img_size(vimgLeft->width(), vimgLeft->height());

            // left intrinsics and distortion
            try
            {
                Math::CameraIntrinsics< double > intrinsics_left = *m_inPortIntrinsicsLeft.get( ts );

                // copy ublas to OpenCV parameters
                D1.create( 1, 8, CV_64F );
				D1.at<double>( 0 ) = static_cast< double >(intrinsics_left.radial_params( 0 ));
				D1.at<double>( 1 ) = static_cast< double >(intrinsics_left.radial_params( 1 ));
				D1.at<double>( 2 ) = static_cast< double >(intrinsics_left.tangential_params( 0 ));
				D1.at<double>( 3 ) = static_cast< double >(intrinsics_left.tangential_params( 1 ));
				D1.at<double>( 4 ) = static_cast< double >(intrinsics_left.radial_params( 2 ));
				D1.at<double>( 5 ) = static_cast< double >(intrinsics_left.radial_params( 3 ));
				D1.at<double>( 6 ) = static_cast< double >(intrinsics_left.radial_params( 4 ));
				D1.at<double>( 7 ) = static_cast< double >(intrinsics_left.radial_params( 5 ));

                M1.create( 3, 3, CV_64F );
                for ( std::size_t i = 0; i < 3; i++ )
					for ( std::size_t j = 0; j < 3; j++ ) {
						// OpenCV vs UbiTrack Intrinsics ..
						if (j == 2) {
							reinterpret_cast< double* >( M1.data + i * M1.step)[ j ]
								= - static_cast< double >( intrinsics_left.matrix( i, j ) );
						} else {
							reinterpret_cast< double* >( M1.data + i * M1.step)[ j ]
								= static_cast< double >( intrinsics_left.matrix( i, j ) );
						}
					}

            } catch( ... )
            {
                UBITRACK_THROW("Error retrieving left camera calibration." );
            }

            // right intrinsics and distortion
            try
            {
                Math::CameraIntrinsics< double > intrinsics_right = *m_inPortIntrinsicsRight.get( ts );

                // copy ublas to OpenCV parameters
                D2.create( 1, 8, CV_64F );
				D2.at<double>( 0 ) = static_cast< double >(intrinsics_right.radial_params( 0 ));
				D2.at<double>( 1 ) = static_cast< double >(intrinsics_right.radial_params( 1 ));
				D2.at<double>( 2 ) = static_cast< double >(intrinsics_right.tangential_params( 0 ));
				D2.at<double>( 3 ) = static_cast< double >(intrinsics_right.tangential_params( 1 ));
				D2.at<double>( 4 ) = static_cast< double >(intrinsics_right.radial_params( 2 ));
				D2.at<double>( 5 ) = static_cast< double >(intrinsics_right.radial_params( 3 ));
				D2.at<double>( 6 ) = static_cast< double >(intrinsics_right.radial_params( 4 ));
				D2.at<double>( 7 ) = static_cast< double >(intrinsics_right.radial_params( 5 ));

                M2.create( 3, 3, CV_64F );
                for ( std::size_t i = 0; i < 3; i++ )
					for ( std::size_t j = 0; j < 3; j++ ) {
						// OpenCV vs UbiTrack Intrinsics ..
						if (j == 2) {
							reinterpret_cast< double* >( M2.data + i * M2.step)[ j ]
								= - static_cast< double >( intrinsics_right.matrix( i, j ) );
						} else {
							reinterpret_cast< double* >( M2.data + i * M2.step)[ j ]
								= static_cast< double >( intrinsics_right.matrix( i, j ) );
						}
					}

            } catch( ... )
            {
                UBITRACK_THROW("Error retrieving right camera calibration." );
            }

            // left2right transform
            // right intrinsics and distortion
            try
            {
                Math::Pose cameraOffset = *m_inPortCameraOffset.get( ts );
                Math::Vector< double, 3 > l2r_translation = cameraOffset.translation();
                Math::Matrix< double, 3, 3 > l2r_rotation;
                cameraOffset.rotation().toMatrix(l2r_rotation);

                // copy ublas to OpenCV parameters
                T.create(3,1, CV_64F);
                for ( std::size_t i = 0; i< 3; ++i )
                    reinterpret_cast< double* >( T.data )[ i ] = static_cast< double >( l2r_translation( i ) );

                R.create( 3, 3, CV_64F );
                for ( std::size_t i = 0; i < 3; i++ )
                    for ( std::size_t j = 0; j < 3; j++ )
                        reinterpret_cast< double* >( R.data + i * R.step)[ j ]
                            = static_cast< double >( l2r_rotation( i, j ) );

            } catch( ... )
            {
                UBITRACK_THROW("Error retrieving left to right transform." );
            }

            Mat R1, P1, R2, P2, Q;
            Rect roi1, roi2;
			try
			{
				stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );
			} catch (cv::Exception const& e) {
				UBITRACK_THROW("Error computing rectification transforms");
			}

			// initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
			// initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
			
			// send QMatrix once
            Math::Matrix< double, 4, 4 > out_Q;
            for ( std::size_t i = 0; i < 3; i++ )
                for ( std::size_t j = 0; j < 3; j++ )
                    out_Q( i, j ) = Q.at<double>(i, j);
            m_outPortQMatrix.send( Measurement::Matrix4x4( ts, out_Q ) );

            // setup stereo match algorithms
            // bm.ndisp = m_ndisparity;
            // bm.winSize = m_windowSize;
            // bm.preset = m_bmFilter;

            // bp.ndisp = m_ndisparity;
            // bp.iters = m_iterations;
            // bp.levels = m_levels;

            // csbp.ndisp = m_ndisparity;
            // csbp.iters = m_iterations;
            // csbp.levels = m_levels;

            // cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

            m_initialized = true;
        }

		// undistort and rectify left and right image
		boost::shared_ptr< Image > imgLeftr( new Image( imgLeft.size().width, imgLeft.size().height, imgLeft.channels(), vimgLeft->depth() ) );
		imgLeftr->set_origin(vimgLeft->origin());
		cv::Mat imgLeftrm( imgLeftr->Mat(), false );

		boost::shared_ptr< Image > imgRightr( new Image( imgRight.size().width, imgRight.size().height, imgRight.channels(), vimgRight->depth() ) );
		imgRightr->set_origin(vimgRight->origin());
        cv::Mat imgRightrm( imgRightr->Mat(), false );

		try {
		    // remap(imgLeft, imgLeftrm, map11, map12, INTER_LINEAR);
		} catch (cv::Exception const& e) {
			std::cerr << e.what() << std::endl;
			UBITRACK_THROW("Error undistorting left image");
		}

		try {
	        // remap(imgRight, imgRightrm, map21, map22, INTER_LINEAR);
		} catch (cv::Exception const& e) {
			std::cerr << e.what() << std::endl;
			UBITRACK_THROW("Error undistorting right image");
		}


        // BM does only support greyscale images
        if ((m_method == BM) && ((imgLeftrm.channels() > 1) || (imgRightrm.channels() > 1))) {
            Mat imgLeftrg, imgRightrg;
            cvtColor(imgLeftrm, imgLeftrg, CV_BGR2GRAY);
            cvtColor(imgRightrm, imgRightrg, CV_BGR2GRAY);
            imgLeftrm = imgLeftrg;
            imgRightrm = imgRightrg;
        }

		// upload images to gpu
        // d_left.upload(imgLeftrm);
        // d_right.upload(imgRightrm);


		boost::shared_ptr< Image > disp_out( new Image( imgLeftrm.size().width, imgLeftrm.size().height, CV_8U ) );
		disp_out->set_origin(imgLeftr->origin());
		cv::Mat disp( disp_out->Mat(), false );

		// if (d_disp.empty()) {
	 //        d_disp.create(imgLeftrm.size(), CV_8U);
		// }

  //       switch (m_method)
  //       {
  //       case BM:
  //           bm(d_left, d_right, d_disp);
  //           break;
  //       case BP: bp(d_left, d_right, d_disp); break;
  //       case CSBP: csbp(d_left, d_right, d_disp); break;
  //       }

		// d_disp.download(disp);

        LOG4CPP_DEBUG( logger, "Done computing Disparity Map." );

        // send results
        // DisparityMap
        m_outPortDisparity.send( ImageMeasurement( ts, disp_out ) );

        m_outPortImageLeft.send( ImageMeasurement( ts, imgLeftr ) );
        m_outPortImageRight.send( ImageMeasurement( ts, imgRightr ) );
    }
};

UBITRACK_REGISTER_COMPONENT( Dataflow::ComponentFactory* const cf )
{
    // for the old pattern
    cf->registerComponent< StereoMatchGPU > ( "StereoMatchGPU" );

}

} } // namespace Ubitrack::MVLVision
