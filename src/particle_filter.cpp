/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(const double x, const double y, const double theta, const double std[]) {
  num_particles = 100;
  particles.resize(num_particles);

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  size_t id = 0;
  for (Particle& particle : particles) {
    particle.id = id++; // id is used nowhere currently but all particle will have an own id at initialization
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0; // It will be overriden so it is not so important here
  }

  is_initialized = true;
}

void ParticleFilter::prediction(const double delta_t, const double std_pos[], 
                                const double velocity, const double yaw_rate) {
  for (Particle& particle : particles) {
    double newX;
    double newY;
    double newTheta;
    if (fabs(yaw_rate) < 1e-8) {
      // If yar rate is close to zero we an suppose the car is moving ahead, so the usual equation of motion can be used.
      const double velocity_dt = velocity * delta_t; // forward calculated, because it will be used twice in next steps
      newX = particle.x + velocity_dt * cos(particle.theta);
      newY = particle.y + velocity_dt * sin(particle.theta);
      newTheta = particle.theta;
    } else {
      // If yar rate is not close to zero an updated equation of motion should be used
      // Some value will be forward calculated, in order to use them in next steps
      const double yaw_rate_dt = yaw_rate * delta_t;
      const double velocity_per_yaw_rate = velocity / yaw_rate;
      newX = particle.x + velocity_per_yaw_rate * (sin(particle.theta + yaw_rate_dt) - sin(particle.theta));
      newY = particle.y + velocity_per_yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate_dt));
      newTheta = particle.theta + yaw_rate_dt;
    }

    // Random Gaussian noise should be given
    std::normal_distribution<double> dist_x(newX, std_pos[0]);
    std::normal_distribution<double> dist_y(newY, std_pos[1]);
    std::normal_distribution<double> dist_theta(newTheta, std_pos[2]);
    
    // Particle state should be updated
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs>& predicted, 
                                     vector<LandmarkObs>& observations) {
  // For each observation the closest landmark position will be assigned to
  for (LandmarkObs& observation : observations) {
    size_t minIndex = 0;
    double minDistance = dist(predicted[minIndex].x, predicted[minIndex].y, observation.x, observation.y);
    for (size_t i = 1; i < predicted.size(); ++i) {
      double currDistance = dist(predicted[i].x, predicted[i].y, observation.x, observation.y);
      if (currDistance < minDistance) {
        minIndex = i;
        minDistance = currDistance;
      }
    }
    observation.id = predicted[minIndex].id;
  }
}

void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  // Precalculate values for Gaussian distribution
  const double stddev_x = std_landmark[0] * std_landmark[0];
  const double stddev_y = std_landmark[1] * std_landmark[1];
  const double normalizer = 1 / sqrt(2 * M_PI * std_landmark[0] * std_landmark[1]);

  // Previous weights will be overwritten, so it can be cleared here
  weights.clear();
  weights.shrink_to_fit();

  for (Particle& particle : particles) {
    const double cosTheta = cos(particle.theta); // It will be used twice so useful to precalculate
    const double sinTheta = sin(particle.theta); // It will be used twice so useful to precalculate
    
    // Transform each observations from car (or particle) coordinate system to map coordinate system
    vector<LandmarkObs> observationsInMapCoordinateSystem;
    const int invalidLandmarkId = std::numeric_limits<int>::max();
    for (const LandmarkObs& observation : observations) {
      const double x_map = cosTheta * observation.x - sinTheta * observation.y + particle.x;
      const double y_map = sinTheta * observation.x + cosTheta * observation.y + particle.y;
      observationsInMapCoordinateSystem.push_back(LandmarkObs{invalidLandmarkId, x_map, y_map}); // Landmark id is irrelevant here
    }

    // Select landmarks which are close enough to our car
    vector<LandmarkObs> closestLandmarks;
    for (const auto& landmark : map_landmarks.landmark_list) {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
        closestLandmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // Associate observations to the closest landmarks
    dataAssociation(closestLandmarks, observationsInMapCoordinateSystem);

    particle.weight = 1.0; // It will be multiplied so the initial value must be 1.0

    // Only for debugging
    // vector<int> associations;
    // vector<double> senseX;
    // vector<double> senseY;

    // For each observation calculate the difference between the observation and its closest landmark
    // The weight multiplier of the particle will be calculated from this difference
    for (const LandmarkObs& observation : observationsInMapCoordinateSystem) {
      // Find the closest landmark which has the desired id
      LandmarkObs closestLandmark;
      size_t j = 0;
      while (j < closestLandmarks.size() && closestLandmarks[j].id != observation.id) {
        j++;
      }
      if (j < closestLandmarks.size()) {
        closestLandmark = closestLandmarks[j];
      } else {
        std::cout << "No matching landmark" << std::endl;
      }

      const double diffX = observation.x - closestLandmark.x;
      const double diffY = observation.y - closestLandmark.y;
      const double currWeight = exp(-0.5 * (diffX * diffX / stddev_x + diffY * diffY / stddev_y)) * normalizer;
      particle.weight *= currWeight;

      // Only for debugging
      // associations.push_back(observation.id);
      // senseX.push_back(observation.x);
      // senseY.push_back(observation.y);
      }

    // Weights are already stored as a particle member, however it is worth to store in an other vector which will be used in the resampling step.
    weights.push_back(particle.weight);

    // Only for debugging
    // SetAssociations(particle, associations, senseX, senseY);
  }
}

void ParticleFilter::resample() {
  std::discrete_distribution<> dist_weights(weights.begin(), weights.end());

  vector<Particle> newParticles(num_particles); // Vector of newly selected particles
  for (Particle& particle : newParticles) {
    particle = particles[dist_weights(gen)];
  }

  particles = newParticles; // Previous particles should be overwritten
  
  // As a first try I used resampling wheel, but using std::discrete_distribution seemed much easier
  // weights.clear();
  // weights.shrink_to_fit();
  // double maxWeight = std::numeric_limits<double>::min();
  // for (const auto& particle : particles) {
  //   weights.push_back(particle.weight);
  //   if (particle.weight > maxWeight) {
  //     maxWeight = particle.weight;
  //   }
  // }

  // std::cout << "Maxweight" << maxWeight << std::endl;

  // const size_t n = weights.size();
  // std::default_random_engine gen;
  // std::uniform_int_distribution<int> uniform_int(0, n - 1);
  // int index = uniform_int(gen);
  // double beta = 0.0;

  // std::uniform_real_distribution<double> uniform_real(0, 2 * maxWeight);

  // vector<Particle> newParticles;

  // for (size_t i = 0; i < n; ++i) {
  //   beta += uniform_real(gen);
  //   while (beta > weights[index]) {
  //     beta -= weights[index];
  //     index = (index + 1) % n;
  //   }
  //   newParticles.push_back(particles[index]);
  // }

  // particles = newParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}