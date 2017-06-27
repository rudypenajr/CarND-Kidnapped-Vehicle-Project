/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO:
	//  1. Set the number of particles.
	//  2. Initialize all particles to first position
	//     (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
	//  3. Add random Gaussian noise to each particle.
	//
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//
	// NOTE: Refer to Lesson 14: Implementation of a Particle Filter, Sub Lesson 4, 5, and 6
	//

	// http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	// 1.
	num_particles = 100;
	weights.resize(num_particles);

	// 2. & 3.
	// x, y, and psi are set in `std`
	// create normal Gaussian distribution for x, y, and psi
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i=0; i < num_particles; ++i) {
		Particle particle = {
			i,
			dist_x(gen),
			dist_y(gen),
			dist_theta(gen),
			1.0 // weight
		};

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	//
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	//
	// NOTE: Refer to Lesson 14: Implementation of a Particle Filter, Sub Lesson 6, 7, and 8

	default_random_engine gen;

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	double threshold = 0.001;

	// equations:
	// x_final = x_initial + (velocity / yaw_rate) * [ sin(yaw_angle + (yaw_rate*dt)) - sin(yaw_angle) ]
	// y_final = y_initial + (velocity / yaw_rate) * [ cos(yaw_angle) - cos(yaw_angle + (yaw_rate*dt)) ]
	// theta_final = yaw_angle + (yaw_rate * dt)
	for (auto &p: particles) {
		if (fabs(yaw_rate) > threshold) {
			p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + (yaw_rate*delta_t)) - sin(p.theta));
			p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + (yaw_rate*delta_t)));
			p.theta = p.theta + delta_t * yaw_rate;
		} else {
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * sin(p.theta);
		}

		// gaussian noise
		normal_distribution<double> dist_x(p.x, std_x);
		normal_distribution<double> dist_y(p.y, std_y);
		normal_distribution<double> dist_theta(p.theta, std_theta);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// Helpful Link: https://discussions.udacity.com/t/implementing-data-association/243745/7
	// for each particle:
	// -> for each observation:
	// --> transform_observation_to_map
	// ---> for each landmark:
	// ----> calc euclidean distance and associate TO particle THE landmark_id w/ THE min_distance
	for (int i=0; i < observations.size(); ++i) {
		double lowest_distance;
		double new_x;
		double new_y;
		LandmarkObs observed_land = observations[i];

		for (int j=0; j < predicted.size(); ++j) {
			LandmarkObs predicted_land = predicted[j];
			double distance = dist(predicted_land.x, predicted_land.y, observed_land.x, observed_land.y);

			if (j == 0) {
				new_x = predicted_land.x;
				new_y = predicted_land.y;
				lowest_distance = distance;
			} else {
				if (distance < lowest_distance) {
					new_x = predicted_land.x;
					new_y = predicted_land.y;
					lowest_distance = distance;
				}
			}

			observations[i].x = new_x;
			observations[i].y = new_y;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution.
	// You can read more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//
	// NOTE: Great Additional Resource: https://discussions.udacity.com/t/kidnapped-vehicle-p3-tips/245197
	// Particles -> Observations -> Landmarks.
	// for particles
	// --> for observations (vehicle coordinates system)
	// ---> for landmarks


	for (int i=0; i < num_particles; ++i) {
		Particle &p = particles[i];

		// initialize weight
		double weight = 1.0;

		for (auto &obs: observations) {
			// observations (vehicle coordinates system)
			// convert to map coordinates
			double obs_x;
			double obs_y;

			// http://planning.cs.uiuc.edu/node99.html
			// x * cos(theta) - y * sin(theta) + p.x
			// x * sin(theta) + y * cos(theta) + p.y
			obs_x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
			obs_y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;

			// Assign each observation to the closest landmark
			Map::single_landmark_s closest_landmark = { 0, 0.0, 0.0 };
			double min_distance_observation_to_landmark = sensor_range;

			for (auto &landmark: map_landmarks.landmark_list) {
				// calculate particle to landmark distance
				double distance_particle_to_landmark = dist(p.x, p.y, landmark.x_f, landmark.y_f);

				if (distance_particle_to_landmark <= sensor_range) {
					double distance_observation_to_landmark = dist(obs_x, obs_y, landmark.x_f, landmark.y_f);

					if (distance_observation_to_landmark < min_distance_observation_to_landmark) {
						min_distance_observation_to_landmark = distance_observation_to_landmark;
						closest_landmark = landmark;
					}
				}
			}

			// multivariate gaussian probaili density function TIME!
			double x_diff = closest_landmark.x_f - obs_x;
			double y_diff = closest_landmark.y_f - obs_y;
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			double x_y_term = ( (x_diff * x_diff) / (2 * std_x * std_x) ) + ( (y_diff * y_diff) / (2 * std_y * std_y) );
			long double w = exp(-0.5 * x_y_term) / (2 * M_PI * std_x * std_y);

			weight = weight * w;
		}

		p.weight = weight;
		weights[i] = weight;
	}

	// weights normalization
	double weights_sum = accumulate(weights.begin(), weights.end(), 0.0);

	for (int i =0; i<num_particles; ++i) {
		particles[i].weight = particles[i].weight / weights_sum;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> d(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for (int i=0; i < num_particles; ++i) {
		int idx = d(gen);
		new_particles.push_back(particles[idx]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
