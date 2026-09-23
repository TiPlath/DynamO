#include <dynamo/systems/gaussianThermostat.hpp>

#include <dynamo/NparticleEventData.hpp>
#include <dynamo/dynamics/dynamics.hpp>
#include <dynamo/particle.hpp>
#include <dynamo/species/species.hpp>
#include <dynamo/units/units.hpp>
#include <magnet/xmlreader.hpp>
#include <magnet/xmlwriter.hpp>

namespace dynamo {

SysGaussian::SysGaussian(const magnet::xml::Node &XML,
                         dynamo::Simulation *sim)
    : System(sim), meanFreeTime(1.0), kT(sim->units.unitEnergy()) {
  dt = std::numeric_limits<float>::infinity();
  SysGaussian::operator<<(XML);
  type = RESCALE;
}

SysGaussian::SysGaussian(dynamo::Simulation *sim, double mft, double targetkT,
                         std::string name)
    : System(sim), meanFreeTime(mft), kT(targetkT) {
  sysName = name;
  type = RESCALE;
}

NEventData SysGaussian::runEvent() {
  ++Sim->eventCount;
  Sim->dynamics->updateAllParticles();

  const double currentkT = Sim->dynamics->getkT();
  if (!(currentkT > 0.0) || !std::isfinite(currentkT)) {
    M_throw() << "Gaussian thermostat encountered invalid current kT: "
              << currentkT;
  }

  NEventData data;
  for (const shared_ptr<Species> &species : Sim->species)
    for (const unsigned long &partID : *species->getRange())
      data.L1partChanges.push_back(
          ParticleEventData(Sim->particles[partID], *species, RESCALE));

  // Global deterministic velocity rescaling. The existing dynamics method
  // handles translational and rotational kinetic energy and Lees-Edwards BCs.
  Sim->dynamics->rescaleSystemKineticEnergy(kT / currentkT);

  // Keep the target COM velocity at zero, consistent with SysRescale.
  Sim->setCOMVelocity();

  dt = getGhostt();
  return data;
}

void SysGaussian::initialise(size_t nID) {
  ID = nID;
  dt = getGhostt();
}

void SysGaussian::operator<<(const magnet::xml::Node &XML) {
  meanFreeTime = XML.getAttribute("MFT").as<double>() * Sim->units.unitTime();
  kT = XML.getAttribute("Temperature").as<double>() * Sim->units.unitEnergy();
  sysName = XML.getAttribute("Name");
}

void SysGaussian::outputXML(magnet::xml::XmlStream &XML) const {
  XML << magnet::xml::tag("System") << magnet::xml::attr("Type") << "Gaussian"
      << magnet::xml::attr("Name") << sysName
      << magnet::xml::attr("MFT")
      << meanFreeTime / Sim->units.unitTime()
      << magnet::xml::attr("Temperature")
      << kT / Sim->units.unitEnergy()
      << magnet::xml::endtag("System");
}

double SysGaussian::getGhostt() const {
  return -meanFreeTime *
         std::log(1.0 - std::uniform_real_distribution<>()(
                            Sim->ranGenerator));
}

} // namespace dynamo
