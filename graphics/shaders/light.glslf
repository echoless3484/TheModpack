#include "depth_utils.glslh"
#include "lighting_common.glslh"
#include "shadow_common.glslh"
#include "environment_lighting.glslh"

#ifndef ENABLE_SSAO
    #define ENABLE_SSAO 1
#endif

in vec2 vertex_coord0_out;
out vec4 color_out;

uniform sampler2D gnormal;
uniform sampler2D gcolor;
uniform sampler2D gdepth;

#if ENABLE_SSAO == 1
    uniform sampler2D texture_ssao;
#endif

uniform sampler2D texture_water_depth;

uniform mat4 mat_view_proj_inverse;

uniform vec4 far_distances;
uniform mat4 mat_world_to_shadow[4];
uniform sampler2DShadow texture_shadow_main_light0;

uniform vec3 camera_position;
uniform float camera_near;
uniform float camera_far;

uniform float main_shadow_factor;

uniform vec3 sun_light_direction;
uniform vec3 sun_light_color;
uniform float sun_light_intensity_global;

uniform vec3 moon_light_direction;
uniform vec3 moon_light_color;
uniform float moon_light_intensity_global;

uniform float shadow_factor;

uniform vec3 sky_color_horizon;
uniform vec3 sky_color_zenith;
uniform vec4 ambient_color_low;
uniform vec4 ambient_color_high;

uniform float fog_density;
uniform vec2 rand_offset; 

uniform float ash_density;
uniform vec3 ash_color;

uniform int is_underwater;
uniform vec3 underwater_color;
uniform int is_ssao_enabled;

uniform vec3 camera_position_world;
uniform vec3 graphics_offset;

// PCF parameters
#define PCF_FILTER_STEP_COUNT 1.5
#define PCF_DIM (PCF_FILTER_STEP_COUNT * 2 + 1)
#define PCF_COUNT (PCF_DIM * PCF_DIM)

float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0;
    return (2.0 * camera_near * camera_far) / (camera_far + camera_near - z * (camera_far - camera_near));
}

// Performs PCF filtering on the shadow map using multiple taps in the filter region.
float PCF_Filter(sampler2DShadow shadow_map, vec2 uv, float z, vec2 filterRadiusUV)
{
    vec2 stepUV = filterRadiusUV;

#if (SHADOW_QUALITY < 1) // Use 4-tap PCF for low quality
    vec4 samples;

    samples.r = texture2DCompare(shadow_map, uv + vec2(1, 0) * stepUV, z);
    samples.g = texture2DCompare(shadow_map, uv + vec2(-1, 0) * stepUV, z);
    samples.b = texture2DCompare(shadow_map, uv + vec2(0, 1) * stepUV, z);
    samples.a = texture2DCompare(shadow_map, uv + vec2(0, -1) * stepUV, z);

    return dot(samples, vec4(0.25));
#elif (SHADOW_QUALITY == 1) // Use 12-tap poisson filtering for medium quality
    float sum = 0;

    for(int i = 0; i < 12; ++i)
    {
        vec2 offset = poisson_disc[i];
        offset *= stepUV;

        sum += texture2DCompare(shadow_map, uv + offset, z);
    }

    return sum / 12.0;
#else
    return PCF_Filter7x7(shadow_map, uv, z, filterRadiusUV.x);
#endif
}

float filter_shadow(sampler2DShadow shadow_map, vec4 shadow_coord0, vec2 filter_width)
{
    return PCF_Filter(shadow_map, shadow_coord0.xy, max(shadow_coord0.z - 0.0002, 0.0), filter_width);
}

float shadow(float linear_depth, vec3 world_position, vec3 normal)
{
    int shadow_index = 3;
    float shadow_mix_factor = 0.0;

    const float blur_threshold = 0.8;

    vec2 filter_width = vec2(2.0) / textureSize(texture_shadow_main_light0, 0);

    vec4 shadow_coord0;
    vec4 shadow_coord1;

    if(linear_depth < far_distances.x)
    {
        shadow_index = 0;
        shadow_mix_factor = clamp(linear_depth / far_distances.x, 0.0, 1.0);
    }
    else if(linear_depth < far_distances.y)
    {
        shadow_index = 1;
        shadow_mix_factor = clamp((linear_depth - far_distances.x) / (far_distances.y - far_distances.x), 0.0, 1.0);
    }
    else if (linear_depth < far_distances.z)
    {
        shadow_index = 2;
        shadow_mix_factor = clamp((linear_depth - far_distances.y) / (far_distances.z - far_distances.y), 0.0, 1.0);
    }
    else if (linear_depth < far_distances.w)
    {
        shadow_index = 3;
    }
    else
    {
        return 1.0;
    }

    shadow_mix_factor = clamp((shadow_mix_factor - blur_threshold) / (1.0 - blur_threshold), 0.0, 1.0);

    float shadow_value0 = 1.0;
    float shadow_value1 = 1.0;

    shadow_coord0 = mat_world_to_shadow[shadow_index] * vec4(world_position, 1.0);
    shadow_coord0 = shadow_coord0 * 0.5 + 0.5;
    shadow_coord0.xy *= 0.5;

    float x = float(shadow_index % 2) * 0.5;
	float y = float(shadow_index < 2 ? 0 : 1) * 0.5;

    shadow_coord0.xy += vec2(x, y);

    shadow_value0 = filter_shadow(texture_shadow_main_light0, shadow_coord0, filter_width);

    if(shadow_mix_factor > 0.0)
    {
        shadow_coord1 = mat_world_to_shadow[shadow_index + 1] * vec4(world_position, 1.0);
        shadow_coord1 = shadow_coord1 * 0.5 + 0.5;
        shadow_coord1.xy *= 0.5;

        float x = float((shadow_index + 1) % 2) * 0.5;
    	float y = float((shadow_index + 1) < 2 ? 0 : 1) * 0.5;

        shadow_coord1.xy += vec2(x, y);

        shadow_value1 = filter_shadow(texture_shadow_main_light0, shadow_coord1, filter_width);
    }

    return mix(shadow_value0, shadow_value1, shadow_mix_factor);
}

float rand(vec2 co) // returns -1 -> +1
{
	return (fract(sin(dot(co.xy + rand_offset, vec2(12.9898,78.233))) * 43758.5453) * 2.0) - 1.0;
}

void main()
{
    // Get info from g-buffer
    float linear_depth = decode_depth_linear(texture(gdepth, vertex_coord0_out).r);
    float depth = DeLinearizeDepth(linear_depth);

    vec3 position = world_pos_from_depth(mat_view_proj_inverse, vertex_coord0_out, depth);
    vec4 normal_light_factor = texture(gnormal, vertex_coord0_out);
    vec3 normal = normalize(normal_light_factor.xyz);
    vec3 color = texture(gcolor, vertex_coord0_out).rgb;

#if ENABLE_SSAO == 1
    float ambient_occlusion = 1.0;
    if(is_ssao_enabled > 0)
    {
        ambient_occlusion = texture(texture_ssao, vertex_coord0_out).x;
    }
#else
    float ambient_occlusion = 1.0;
#endif

    float light_factor = normal_light_factor.a;

    float shadow_value = shadow(linear_depth, position, normal) * ambient_occlusion;
    shadow_value = mix(1.0, shadow_value, main_shadow_factor);

    if(light_factor < 0.25)
    {
        light_factor = 0.0;
    }
    else if(light_factor < 0.75)
    {
        light_factor = 0.25;
    }

    // Ambient
    ambient_occlusion = mix(1.0, ambient_occlusion, light_factor);
    vec3 ambient = ambient_color(normal, ambient_color_low.rgb, ambient_color_high.rgb) * ambient_occlusion;
    ambient = mix(mix(ambient_color_low.rgb, ambient_color_high.rgb, 0.5), ambient, light_factor);

    // Specular
    vec3 camera_to_fragment = position - camera_position;
    vec3 eye = normalize(camera_to_fragment);

    vec3 surface_color = vec3(0);

    float light_filter_factor = clamp(1.0 / exp(-(position.y - graphics_offset.y) * underwater_fog_density * 0.1), 0.0, 1.0);
    vec3 light_filter = mix(underwater_color, vec3(1.0), light_filter_factor);

    float roughness = 0.8;

    // Sun diffuse
    float sun_light_intensity = max(0.0, -dot(sun_light_direction, normal));
    sun_light_intensity = mix(1.0, sun_light_intensity * shadow_value, light_factor) * sun_light_intensity_global;
    vec3 sun_contribution = brdf(sun_light_color * light_filter, sun_light_direction, color, vec3(0.02), normal, eye, roughness, light_factor) * sun_light_intensity;

    // Moon diffuse
    float moon_light_intensity = max(0.0, -dot(moon_light_direction, normal));
    moon_light_intensity = mix(1.0, moon_light_intensity * shadow_value, light_factor) * moon_light_intensity_global;
    vec3 moon_contribution = brdf(moon_light_color * light_filter, moon_light_direction, color, vec3(0.02), normal, eye, roughness, light_factor) * moon_light_intensity;
 
    surface_color = (color * ambient) + sun_contribution + moon_contribution;

    surface_color = max(vec3(0), surface_color);

    // Determine whether world position is below waterline
    float this_fog_density = fog_density * 0.25;
    vec3 fog_color = sky_color(eye, sky_color_horizon, sky_color_zenith);
    
    float water_depth_linear = decode_depth_linear(texture(texture_water_depth, vertex_coord0_out).r);
    float water_depth = DeLinearizeDepth(water_depth_linear);

    if(eye.y < 0.0 && is_underwater == 1)
    {
        water_depth = 1.0;
    }

    // Fog
    if(textureSize(texture_water_depth, 0).x > 32.0)
    { 
        surface_color = apply_fog(surface_color, fog_color, underwater_color, this_fog_density, linear_depth, water_depth_linear, water_depth, rand(gl_FragCoord.xy), is_underwater, 50.0);
    }

    //Ash
    {
        float air_distance = min(linear_depth, water_depth_linear);
        float depth_factor = 1.0 - (1.0 / exp(air_distance));
        surface_color = mix(surface_color, ash_color, depth_factor * ash_density);
    }

    color_out = vec4(surface_color, 1);
}
