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
 * @ingroup vision_components
 * @file
 * Computes the intrinsic matrices and the distortion coefficients for a stereo hmd including the stereo-transform
 * using the all-in-one opencv calibration method
 *
 * @author Ulrich Eck <ueck@net-labs.de>
 */

//std
#include <vector> 
#include <numeric>
#include <algorithm>

//opencv includes
#include "opencv/cv.h"


// Boost (for data preperation)
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>

// Ubitrack
#include <utUtil/OS.h>
#include <utMath/Vector.h>
#include <utMath/Matrix.h>
#include <utMath/CameraIntrinsics.h>
#include <utDataflow/TriggerComponent.h>
#include <utDataflow/ExpansionInPort.h>
#include <utDataflow/TriggerOutPort.h>
#include <utDataflow/PullConsumer.h>
#include <utDataflow/PushSupplier.h>
#include <utDataflow/ComponentFactory.h>
#include <utMeasurement/Measurement.h>

#include <log4cpp/Category.hh>
static log4cpp::Category& logger( log4cpp::Category::getInstance( "Ubitrack.MVLVision.StereoCameraCalibration" ) );


namespace Ubitrack { namespace MVLVision {

class StereoCameraCalibration
	: public Dataflow::TriggerComponent
{
protected:
	
	///shortcut for the type of 2d measurements
	typedef std::vector < Math::Vector< double, 2 > > vector_2d_type;
	
	///shortcut for the type of 3d measurements
	typedef std::vector < Math::Vector< double, 3 > > vector_3d_type;
	
	/** the intrinsic parameters, estimated so far */
    Math::CameraIntrinsics< double > m_camIntrinsics[2];
    Math::Pose m_transform;

	/** signs constraints for the calibration, 
	please have a look at the OpenCV Doku for explanations */
	int m_flags;
	
	/** Input port of the component for 2D measurements. */
    Dataflow::ExpansionInPort< vector_2d_type > m_inPort2DLeft;
    Dataflow::ExpansionInPort< vector_2d_type > m_inPort2DRight;

	/** Input port of the component for 3D measurements. */
	Dataflow::ExpansionInPort< vector_3d_type > m_inPort3D;

	/** Output port of the component providing the camera intrinsic parameters. */
    Dataflow::PushSupplier< Measurement::CameraIntrinsics > m_intrPortLeft;
    Dataflow::PushSupplier< Measurement::CameraIntrinsics > m_intrPortRight;
    Dataflow::TriggerOutPort< Measurement::Pose > m_transformPort;

	/** signs if the thread is running. */
	boost::mutex m_mutexThread;
	
	/** thread performing camera calibration in the background. */
	boost::scoped_ptr< boost::thread > m_pThread;
	
public:
	/**
	 * Standard component constructor.
	 *
	 * @param sName Unique name of the component.
	 * @param cfg ComponentConfiguration containing all configuration.
	 */
    StereoCameraCalibration( const std::string& sName, boost::shared_ptr< Graph::UTQLSubgraph > pCfg )
		: Dataflow::TriggerComponent( sName, pCfg )
		, m_flags( CV_CALIB_RATIONAL_MODEL ) // for using 6 instead of 3 distortion parameters
        , m_inPort2DLeft( "Points2DLeft", *this )
        , m_inPort2DRight( "Points2DRight", *this )
        , m_inPort3D( "Points3D", *this )
        , m_intrPortLeft( "CameraIntrinsicsLeft", *this )
        , m_intrPortRight( "CameraIntrinsicsRight", *this )
        , m_transformPort( "CameraOffset", *this )
        , m_mutexThread( )
    {
		
		// look for some flags which can be specified...
		if ( pCfg->m_DataflowAttributes.hasAttribute( "fixPrincipalPoint" ) ) // fixes principal point to centre of image plane
			if( pCfg->m_DataflowAttributes.getAttributeString( "fixPrincipalPoint" ) == "true" )
				m_flags |= CV_CALIB_FIX_PRINCIPAL_POINT;
				
		if ( pCfg->m_DataflowAttributes.hasAttribute( "noTangentialDistortion" ) ) // assumes no tangential distortion
			if( pCfg->m_DataflowAttributes.getAttributeString( "noTangentialDistortion" ) == "true" )
				m_flags |= CV_CALIB_ZERO_TANGENT_DIST;
		
		if ( pCfg->m_DataflowAttributes.hasAttribute( "fixAspectRatio" ) ) // fixes aspect ratio, such that fx/fy
			if( pCfg->m_DataflowAttributes.getAttributeString( "fixAspectRatio" ) == "true" )
				m_flags |= CV_CALIB_FIX_ASPECT_RATIO;
		
		///@todo add some other flags, that are possible with newer versions of OpenCV
    }

	/** Method that computes the result. */
	void compute( Measurement::Timestamp t )
	{
		if( hasNewPush() )		
		{
			if( !boost::mutex::scoped_try_lock ( m_mutexThread ) )
			{
				LOG4CPP_WARN( logger, "Cannot perform camera calibration, thread is still busy, sending old calibration data." );
				return;
			}
			else
			{
                const std::vector< vector_2d_type >& points2DLeft = *m_inPort2DLeft.get();
                const std::vector< vector_2d_type >& points2DRight = *m_inPort2DRight.get();
                const std::vector< vector_3d_type >& points3D = *m_inPort3D.get();

                m_pThread.reset( new boost::thread( boost::bind( &StereoCameraCalibration::computeIntrinsic, this, points3D, points2DLeft, points2DRight )));
			}
		}
			
        m_intrPortLeft.send( Measurement::CameraIntrinsics ( t, m_camIntrinsics[0] ) );
        m_intrPortRight.send( Measurement::CameraIntrinsics ( t, m_camIntrinsics[1] ) );
        m_transformPort.send( Measurement::Pose ( t, m_transform ) );
    }

    void computeIntrinsic( const std::vector< vector_3d_type > points3D, const std::vector< vector_2d_type > points2DLeft, const std::vector< vector_2d_type > points2DRight )
	{
		boost::mutex::scoped_lock lock( m_mutexThread );
        const std::size_t m_values = points2DLeft.size();

        if (m_values != points2DRight.size())
        {
            LOG4CPP_ERROR( logger, "Cannot perform camera calibration, left and right samples need to have the same size." );
            return;
        }

		
        if( m_values < 2 )
		{
			// UBITRACK_THROW( "Cannot perform camera calibration, need at lest two different views." );
			LOG4CPP_ERROR( logger, "Cannot perform camera calibration, need at lest two different views." );
			return;
		}
			
		
        if( m_values != points3D.size() )
		{
			// UBITRACK_THROW( "Cannot perform camera calibration, number of views in 2D does not match number of 3D grids." );
			LOG4CPP_ERROR( logger, "Cannot perform camera calibration, number of views in 2D does not match number of 3D grids." );
			return;
		}	

		//count the number of available correspondences
        boost::scoped_array< int > chessNumber( new int[ m_values ] );
		
		
		std::vector< std::size_t > num_points;
        num_points.reserve( m_values );
		
		for( std::size_t i( 0 ); i < m_values; ++i )
		{
            const std::size_t n2Dl = points2DLeft.at( i ).size();
            const std::size_t n2Dr = points2DRight.at( i ).size();
            const std::size_t n3D = points3D.at( i ).size();
            if (( n2Dl != n3D ) || (n2Dr != n3D))
			{
				// UBITRACK_THROW( "Cannot perform camera calibration, number of corresponding 2D/3D measurements does not match." );
				LOG4CPP_ERROR( logger, "Cannot perform camera calibration, number of corresponding 2D/3D measurements does not match." );
				return;
			}
				
            num_points.push_back( n2Dl );
            chessNumber[ i ] = static_cast< int > ( n2Dl );
		}
		
        const std::size_t sum2D3D = std::accumulate( num_points.begin(), num_points.end(), 0 );
			
		
		// everything is fine so far, prepare copying the data
        boost::scoped_array< float > imgPointsL( new float[2*sum2D3D] );
        boost::scoped_array< float > imgPointsR( new float[2*sum2D3D] );
        boost::scoped_array< float > objPoints( new float[3*sum2D3D] );
		

		// copy image corners to big array
		for( std::size_t i ( 0 ); i < m_values; ++i )
		{
            const std::size_t corners2DL = points2DLeft.at( i ).size();
            for( std::size_t j ( 0 ) ; j < corners2DL; ++j )
			{
                imgPointsL[2*i*corners2DL + 2*j] = static_cast< float > ( points2DLeft.at( i ).at( j )( 0 ) );
                imgPointsL[2*i*corners2DL + 2*j + 1] = static_cast< float> ( points2DLeft.at( i ).at( j )( 1 ) );
			}
			
            const std::size_t corners2DR = points2DRight.at( i ).size();
            for( std::size_t j ( 0 ) ; j < corners2DR; ++j )
            {
                imgPointsR[2*i*corners2DR + 2*j] = static_cast< float > ( points2DRight.at( i ).at( j )( 0 ) );
                imgPointsR[2*i*corners2DR + 2*j + 1] = static_cast< float> ( points2DRight.at( i ).at( j )( 1 ) );
            }

            const std::size_t corners3D = points3D.at( i ).size();
			for( std::size_t j ( 0 ) ; j < corners3D; ++j )
			{
				objPoints[3*i*corners3D + 3*j] = static_cast< float > ( points3D.at( i ).at( j )( 0 ) );
				objPoints[3*i*corners3D + 3*j + 1] = static_cast< float> ( points3D.at( i ).at( j )( 1 ) );
				objPoints[3*i*corners3D + 3*j + 2] = static_cast< float > ( points3D.at( i ).at( j )( 2 ) );
			}
		}

        CvMat object_points = cvMat ( sum2D3D, 3, CV_32F, objPoints.get() );
        CvMat image_points_left = cvMat ( sum2D3D, 2, CV_32F, imgPointsL.get() );
        CvMat image_points_right = cvMat ( sum2D3D, 2, CV_32F, imgPointsR.get() );
        CvMat point_counts = cvMat( 1, m_values, CV_32S, chessNumber.get() );

        float intrValL[9];
        float intrValR[9];
        CvMat intrinsic_matrix_left = cvMat ( 3, 3, CV_32FC1, intrValL );
        CvMat intrinsic_matrix_right = cvMat ( 3, 3, CV_32FC1, intrValR );

        float disValL[8];
        float disValR[8];
        CvMat distortion_coeffs_left = cvMat( 8, 1, CV_32FC1, disValL );
        CvMat distortion_coeffs_right = cvMat( 8, 1, CV_32FC1, disValR );

        float extrRot[9];
        CvMat extrinsic_rotation = cvMat ( 3, 3, CV_32FC1, extrRot );

        float extrTrans[3];
        CvMat extrinsic_translation = cvMat( 3, 1, CV_32FC1, extrTrans );

        int flags = m_flags | CV_CALIB_SAME_FOCAL_LENGTH | CV_CALIB_RATIONAL_MODEL |
                              CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5;

        double rms;

		try
		{

            rms = cvStereoCalibrate(&object_points,
                                           &image_points_left,
                                           &image_points_right,
                                           &point_counts,
                                           &intrinsic_matrix_left,
                                           &distortion_coeffs_left,
                                           &intrinsic_matrix_right,
                                           &distortion_coeffs_right,
                                           cvSize( 640, 480 ),
                                           &extrinsic_rotation,
                                           &extrinsic_translation,
                                           NULL,
                                           NULL,
                                           cv::TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                                           flags
                                           );
		}
		catch( const std::exception & e )
		{
			LOG4CPP_ERROR( logger, "Cannot perform camera calibration, error in OpenCV function call.\n" << e.what() );
			return;
		}

        Measurement::Timestamp ts = Measurement::now();

		///@todo check if the paramrers should not be flipped, as it is done at some other places.
        const Math::Vector< double, 2 > tangentialL( disValL[ 2 ], disValL[ 3 ] );
        Math::Vector< double, 6 > radialL;
        radialL( 0 ) = disValL[ 0 ];
        radialL( 1 ) = disValL[ 1 ];
        radialL( 2 ) = disValL[ 4 ];
        radialL( 3 ) = disValL[ 5 ];
        radialL( 4 ) = disValL[ 6 ];
        radialL( 5 ) = disValL[ 7 ];
		
        Math::Matrix< double, 3, 3 > intrinsicL;
        intrinsicL( 0, 0 ) = static_cast< double >( intrValL[0] );
        intrinsicL( 0, 1 ) = static_cast< double >( intrValL[1] );
        intrinsicL( 0, 2 ) = -static_cast< double >( intrValL[2] );
        intrinsicL( 1, 0 ) = static_cast< double >( intrValL[3] );
        intrinsicL( 1, 1 ) = static_cast< double >( intrValL[4] );
        intrinsicL( 1, 2 ) = -static_cast< double >( intrValL[5] );
        intrinsicL( 2, 0 ) = 0.0;
        intrinsicL( 2, 1 ) = 0.0;
        intrinsicL( 2, 2 ) = -1.0;

        m_camIntrinsics[0] = Math::CameraIntrinsics< double > ( intrinsicL, radialL, tangentialL );
        m_intrPortLeft.send( Measurement::CameraIntrinsics ( ts, m_camIntrinsics[0] ) );


        ///@todo check if the paramrers should not be flipped, as it is done at some other places.
        const Math::Vector< double, 2 > tangentialR( disValR[ 2 ], disValR[ 3 ] );
        Math::Vector< double, 6 > radialR;
        radialR( 0 ) = disValR[ 0 ];
        radialR( 1 ) = disValR[ 1 ];
        radialR( 2 ) = disValR[ 4 ];
        radialR( 3 ) = disValR[ 5 ];
        radialR( 4 ) = disValR[ 6 ];
        radialR( 5 ) = disValR[ 7 ];

        Math::Matrix< double, 3, 3 > intrinsicR;
        intrinsicR( 0, 0 ) = static_cast< double >( intrValR[0] );
        intrinsicR( 0, 1 ) = static_cast< double >( intrValR[1] );
        intrinsicR( 0, 2 ) = -static_cast< double >( intrValR[2] );
        intrinsicR( 1, 0 ) = static_cast< double >( intrValR[3] );
        intrinsicR( 1, 1 ) = static_cast< double >( intrValR[4] );
        intrinsicR( 1, 2 ) = -static_cast< double >( intrValR[5] );
        intrinsicR( 2, 0 ) = 0.0;
        intrinsicR( 2, 1 ) = 0.0;
        intrinsicR( 2, 2 ) = -1.0;

        m_camIntrinsics[1] = Math::CameraIntrinsics< double > ( intrinsicR, radialR, tangentialR );
        m_intrPortRight.send( Measurement::CameraIntrinsics ( ts, m_camIntrinsics[1] ) );

        Math::Vector< double, 3 > trans;
        trans( 0 ) = extrTrans[ 0 ];
        trans( 1 ) = extrTrans[ 1 ];
        trans( 2 ) = extrTrans[ 2 ];

        Math::Matrix< double, 3, 3 > rot;
        rot( 0, 0 ) = static_cast< double >( intrValR[0] );
        rot( 0, 1 ) = static_cast< double >( intrValR[1] );
        rot( 0, 2 ) = static_cast< double >( intrValR[2] );
        rot( 1, 0 ) = static_cast< double >( intrValR[3] );
        rot( 1, 1 ) = static_cast< double >( intrValR[4] );
        rot( 1, 2 ) = static_cast< double >( intrValR[5] );
        rot( 2, 0 ) = static_cast< double >( intrValR[6] );
        rot( 2, 1 ) = static_cast< double >( intrValR[7] );
        rot( 2, 2 ) = static_cast< double >( intrValR[8] );

        m_transform = Math::Pose(Math::Quaternion(rot), trans);
        m_transformPort.send( Measurement::Pose ( ts, m_transform ) );

        LOG4CPP_INFO( logger, "Finished camera calibration using " << m_values << " views (RMS Error: " << rms << ")." );
    }
};


UBITRACK_REGISTER_COMPONENT( Dataflow::ComponentFactory* const cf ) {
    cf->registerComponent< StereoCameraCalibration > ( "StereoCameraCalibrationCV" );
}

} } // namespace Ubitrack::MVLVision

