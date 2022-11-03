FROM carlasim/carla:0.9.12 as carla

USER root 
RUN apt-get update \
    && apt-get install -y curl libomp5 xdg-user-dirs

# We want to make sure Town06 is imported and ready to use for CARLA.
USER carla
WORKDIR /home/carla

RUN curl -sSL https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.12.tar.gz \
  | tar --keep-newer-files -xvz


RUN SDL_VIDEODRIVER=""
CMD unset SDL_VIDEODRIVER; bash ./CarlaUE4.sh -vulkan -RenderOffScreen -nosound
