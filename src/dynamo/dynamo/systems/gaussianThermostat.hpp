#pragma once

#include <dynamo/simulation.hpp>
#include <dynamo/systems/system.hpp>

namespace dynamo {

/**
 * \brief Event-driven Gaussian-style thermostat.
 *
 * At Poisson-distributed system events, all translational and rotational
 * velocities are rescaled so that the instantaneous kinetic temperature
 * equals the requested target kT. This is a discrete global velocity
 * rescaling thermostat; it is not the continuous Gaussian isokinetic
 * multiplier formulation.
 */
class SysGaussian : public System {
public:
  SysGaussian(const magnet::xml::Node &, dynamo::Simulation *);
  SysGaussian(dynamo::Simulation *, double meanFreeTime, double kT,
              std::string name);

  virtual NEventData runEvent();
  virtual void initialise(size_t);
  virtual void operator<<(const magnet::xml::Node &);

  double getTemperature() const { return kT; }
  void setTemperature(double value) { kT = value; }

protected:
  virtual void outputXML(magnet::xml::XmlStream &) const;
  double getGhostt() const;

  double meanFreeTime;
  double kT;
};

} // namespace dynamo
