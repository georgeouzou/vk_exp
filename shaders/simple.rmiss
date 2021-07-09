#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main()
{	
    float t = float(gl_LaunchIDEXT.y) / float(gl_LaunchSizeEXT.y);
    float darken = 0.9;
    payload.color.rgb = darken * mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);
    payload.color.a = 1.0;
}
