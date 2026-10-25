# Upkie attribution

Robot design, URDFs and meshes are the work of the [Upkie project and its
contributors](https://github.com/upkie/upkie_description), vendored at commit
94735fbe6137276a41de0ff4cc04d2e533fa9e33. SOURCE.json records the original file
hashes. UPSTREAM_README.md retains the upstream citation and credits.

## Licenses

Upkie model files and mjbots meshes retain Apache-2.0. See [LICENSE](LICENSE).
The tire mesh is adapted from [Car Rim Collection Set 4](https://www.thingiverse.com/thing:3014733)
by bramesh501 under CC BY 4.0. Upkie combined Rim_13.STL and Tyre_model_1.STL
and scaled the outer diameter to 100 mm. The original tire credit remains
in assets/upkie/meshes/wheel_tire/README.md. A license copy is in
[CC-BY-4.0.txt](CC-BY-4.0.txt). The repository's MIT license does not replace
these third-party terms.

## Adaptation

The vendored assets are unchanged. PH-Balance constructs a MuJoCo model with
a floating base, active hip/knee joints, ground contact and source actuator
limits. It resolves mesh paths, omits zero-volume visuals and regularizes
zero virtual-frame inertias. Only the torso link is varied in the mass study.

PH-Balance adds control, sensor simulation, estimation, experiments and
presentation. These are simulation adaptations, not behavior validated by
the original robot authors. No endorsement by Upkie, mjbots or bramesh501
is implied. The demo credits Upkie and the tire author on screen.

For academic use, retain the original citation in UPSTREAM_README.md and
identify this work as an adapted floating-base simulation.
