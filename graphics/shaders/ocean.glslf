#include "depth_utils.glslh"
#include "lighting_common.glslh"
#include "ocean_common.glslh"

in float log_z;
in vec3 world_position_out;
in vec3 view_position_out;
flat in vec3 normal_out;
flat in vec3 view_normal_out;

uniform vec3 camera_position;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform float light_intensity_global;

uniform vec3 sky_color_horizon;
uniform vec3 sky_color_zenith;

uniform vec3 underwater_color;

uniform sampler2D texture_color;
uniform sampler2D texture_depth;
uniform vec2 screen_size;

uniform sampler2D texture_noise0;
uniform sampler2D texture_noise1;
uniform float noise_scroll_timer;

uniform sampler2D texture_wave_height_scale;
uniform float world_to_texture_scale;
uniform vec3 world_offset; // offset for graphics, used in sampling height scale texture 

uniform mat4 mat_proj_to_pixel;
uniform mat4 mat_view;
uniform mat4 mat_view_proj;

uniform float fog_density;

uniform float ash_density;
uniform vec3 ash_color;

uniform vec2 rand_offset;

uniform int is_underwater;

out vec4 color_out;

#ifndef ENABLE_REFLECTION
    #define ENABLE_REFLECTION 1
#endif

float distanceSquared(vec2 A, vec2 B) {
    A -= B;
    return dot(A, A);
}

vec3 computeClipInfo(float zn, float zf)
{
    return vec3(zn * zf, zn - zf, zf);
}

float reconstructCSZ(float depth, vec3 clip_info)
{
    return clip_info[0] / (depth * clip_info[1] + clip_info[2]);
}

void swap(inout float a, inout float b) {
     float temp = a;
     a = b;
     b = temp;
}

float fresnel(float spec_color, float intensity)
{
    // Schlick fresnel approximation
    return spec_color + (1.0 - spec_color) * pow((1.0 - intensity), 5);
}

vec3 sky_color(vec3 normal)
{
    float angle_factor = dot(normal, vec3(0, 1, 0)) * 0.5 + 0.5;
    return mix(sky_color_horizon, sky_color_zenith, angle_factor).rgb;
}

vec3 apply_fog(vec3 surface_color, vec3 fog_color, float fog_density, float distance_to_fragment)
{
    highp float depth_factor = 1.0 / exp(distance_to_fragment * fog_density);
    depth_factor = clamp(depth_factor, 0.0, 1.0);

    return mix(fog_color, surface_color, depth_factor);
}

float Noise(vec2 n,float x)
{
    n+=x;
    return fract(sin(dot(n.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise_step1(vec2 uv,float n){
    float a = 1.0;
    float b = 2.0;
    float c = -12.0;
    float t = 1.0;
    return (1.0/(a*4.0+b*4.0-c))*(
        Noise(uv + vec2(-1.0, -1.0) * t, n) * a +
        Noise(uv + vec2( 0.0, -1.0) * t, n) * b +
        Noise(uv + vec2( 1.0, -1.0) * t, n) * a +
        Noise(uv + vec2(-1.0,  0.0) * t, n) * b +
        Noise(uv + vec2( 0.0,  0.0) * t, n) * c +
        Noise(uv + vec2( 1.0,  0.0) * t, n) * b +
        Noise(uv + vec2(-1.0,  1.0) * t, n) * a +
        Noise(uv + vec2( 0.0,  1.0) * t, n) * b +
        Noise(uv + vec2( 1.0,  1.0) * t, n) * a +
        0.0);
}

float rand(vec2 uv) // returns 0 -> +1
{
    float n = 0.07;// * (fract(rand_offset.x) + 1.0);

    float a = 1.0;
    float b = 2.0;
    float c = -2.0;
    float t = 1.0;
    return (4.0/(a*4.0+b*4.0-c))*(
        noise_step1(uv + vec2(-1.0, -1.0) * t, n) * a +
        noise_step1(uv + vec2( 0.0, -1.0) * t, n) * b +
        noise_step1(uv + vec2( 1.0, -1.0) * t, n) * a +
        noise_step1(uv + vec2(-1.0,  0.0) * t, n) * b +
        noise_step1(uv + vec2( 0.0,  0.0) * t, n) * c +
        noise_step1(uv + vec2( 1.0,  0.0) * t, n) * b +
        noise_step1(uv + vec2(-1.0,  1.0) * t, n) * a +
        noise_step1(uv + vec2( 0.0,  1.0) * t, n) * b +
        noise_step1(uv + vec2( 1.0,  1.0) * t, n) * a +
        0.0);
}

/**
    Screen-space ray tracing from 'Efficient GPU Screen-Space Ray Tracing' by Morgan McGuire and Michael Mara.

    \param csOrigin Camera-space ray origin, which must be
    within the view volume and must have z < -0.01 and project within the valid screen rectangle

    \param csDirection Unit length camera-space ray direction

    \param projectToPixelMatrix A projection matrix that maps to pixel coordinates (not [-1, +1] normalized device coordinates)

    \param csZBuffer The depth or camera-space Z buffer, depending on the value of \a csZBufferIsHyperbolic

    \param csZBufferSize Dimensions of csZBuffer

    \param csZThickness Camera space thickness to ascribe to each pixel in the depth buffer

    \param csZBufferIsHyperbolic True if csZBuffer is an OpenGL depth buffer, false (faster) if
     csZBuffer contains (negative) "linear" camera space z values. Const so that the compiler can evaluate the branch based on it at compile time

    \param clipInfo See G3D::Camera documentation

    \param nearPlaneZ Negative number

    \param stride Step in horizontal or vertical pixels between samples. This is a float
     because integer math is slow on GPUs, but should be set to an integer >= 1

    \param jitterFraction  Number between 0 and 1 for how far to bump the ray in stride units
      to conceal banding artifacts

    \param maxSteps Maximum number of iterations. Higher gives better images but may be slow

    \param maxRayTraceDistance Maximum camera-space distance to trace before returning a miss

    \param hitPixel Pixel coordinates of the first intersection with the scene

    \param csHitPoint Camera space location of the ray hit

    Single-layer

 */
bool traceScreenSpaceRay1
   (vec3          csOrigin,
    vec3         csDirection,
    mat4          projectToPixelMatrix,
    sampler2D       csZBuffer,
    vec2          csZBufferSize,
    float           csZThickness,
    float           nearPlaneZ,
    float			stride,
    float           jitterFraction,
    float           maxSteps,
    in float        maxRayTraceDistance,
    out vec2      hitPixel,
	out vec3		csHitPoint,
    out vec3 user_info) {

    // Clip ray to a near plane in 3D (doesn't have to be *the* near plane, although that would be a good idea)
    float rayLength = ((csOrigin.z + csDirection.z * maxRayTraceDistance) > nearPlaneZ) ?
                        (nearPlaneZ - csOrigin.z) / csDirection.z :
                        maxRayTraceDistance;
	vec3 csEndPoint = csDirection * rayLength + csOrigin;

    // Project into screen space
    vec4 H0 = projectToPixelMatrix * vec4(csOrigin, 1.0);
    vec4 H1 = projectToPixelMatrix * vec4(csEndPoint, 1.0);

    // There are a lot of divisions by w that can be turned into multiplications
    // at some minor precision loss...and we need to interpolate these 1/w values
    // anyway.
    //
    // Because the caller was required to clip to the near plane,
    // this homogeneous division (projecting from 4D to 2D) is guaranteed
    // to succeed.
    float k0 = 1.0 / H0.w;
    float k1 = 1.0 / H1.w;

    // Switch the original points to values that interpolate linearly in 2D
    vec3 Q0 = csOrigin * k0;
    vec3 Q1 = csEndPoint * k1;

	// Screen-space endpoints
    vec2 P0 = H0.xy * k0;
    vec2 P1 = H1.xy * k1;

    // [Optional clipping to frustum sides here]

    // Initialize to off screen
    hitPixel = vec2(-1.0, -1.0);

    // If the line is degenerate, make it cover at least one pixel
    // to avoid handling zero-pixel extent as a special case later
    P1 += vec2((distanceSquared(P0, P1) < 0.0001) ? 0.01 : 0.0);

    vec2 delta = P1 - P0;

    // Permute so that the primary iteration is in x to reduce
    // large branches later
    bool permute = false;
	if (abs(delta.x) < abs(delta.y)) {
		// More-vertical line. Create a permutation that swaps x and y in the output
		permute = true;

        // Directly swizzle the inputs
		delta = delta.yx;
		P1 = P1.yx;
		P0 = P0.yx;
	}

	// From now on, "x" is the primary iteration direction and "y" is the secondary one

    float stepDirection = sign(delta.x);
    float invdx = stepDirection / delta.x;
    vec2 dP = vec2(stepDirection, invdx * delta.y);

    // Track the derivatives of Q and k
    vec3 dQ = (Q1 - Q0) * invdx;
    float dk = (k1 - k0) * invdx;

    // Because we test 1/2 a texel forward along the ray, on the very last iteration
    // the interpolation can go past the end of the ray. Use these bounds to clamp it.
    float zMin = min(csEndPoint.z, csOrigin.z);
    float zMax = max(csEndPoint.z, csOrigin.z);

    // Scale derivatives by the desired pixel stride
    // float cb_strideZCutoff = 400;
    // float strideScale = 1.0f - clamp(min(1.0f, -csOrigin.z / cb_strideZCutoff), 0.0, 1.0);
    // stride = 1.0f + strideScale * stride;

	dP *= stride;
    dQ *= stride;
    dk *= stride;

    // Offset the starting values by the jitter fraction
	P0 += dP * jitterFraction;
    Q0 += dQ * jitterFraction;
    k0 += dk * jitterFraction;

	// Slide P from P0 to P1, (now-homogeneous) Q from Q0 to Q1, and k from k0 to k1
    vec3 Q = Q0;
    float k = k0;

	// We track the ray depth at +/- 1/2 pixel to treat pixels as clip-space solid
	// voxels. Because the depth at -1/2 for a given pixel will be the same as at
	// +1/2 for the previous iteration, we actually only have to compute one value
	// per iteration.
	float prevZMaxEstimate = csOrigin.z;
    float stepCount = 0.0;
    float rayZMax = prevZMaxEstimate;
    float rayZMin = prevZMaxEstimate;
    float sceneZ = rayZMax + 1e4;

    // P1.x is never modified after this point, so pre-scale it by
    // the step direction for a signed comparison
    float end = P1.x * stepDirection;

    // We only advance the z field of Q in the inner loop, since
    // Q.xy is never used until after the loop terminates.

    vec2 P;
	for (P = P0;
        ((P.x * stepDirection) <= end) &&
        (stepCount < maxSteps) &&
        ((rayZMax < sceneZ - csZThickness) || (rayZMin > sceneZ)) &&
        (sceneZ != 0.0);
        P += dP, Q.z += dQ.z, k += dk, stepCount += 1.0) {

        // The depth range that the ray covers within this loop
        // iteration.  Assume that the ray is moving in increasing z
        // and swap if backwards.  Because one end of the interval is
        // shared between adjacent iterations, we track the previous
        // value and then swap as needed to ensure correct ordering
        rayZMin = prevZMaxEstimate;

        // Compute the value at 1/2 pixel into the future
        rayZMax = (dQ.z * 0.5 + Q.z) / (dk * 0.5 + k);
        rayZMax = clamp(rayZMax, zMin, zMax);
		prevZMaxEstimate = rayZMax;

        if (rayZMin > rayZMax) { swap(rayZMin, rayZMax); }

        // Camera-space z of the background
		hitPixel = permute ? P.yx : P;
        sceneZ = texelFetch(csZBuffer, ivec2(hitPixel), 0).r;

    } // pixel on ray

    // Undo the last increment, which ran after the test variables
    // were set up.
    P -= dP; Q.z -= dQ.z; k -= dk; stepCount -= 1.0;
    bool hit = (rayZMax >= sceneZ - csZThickness) && (rayZMin <= sceneZ);

   // If using non-unit stride and we hit a depth surface...
   if ((stride > 1) && hit)
   {
       // Refine the hit point within the last large-stride step

       // Retreat one whole stride step from the previous loop so that
       // we can re-run that iteration at finer scale
       P -= dP; Q.z -= dQ.z; k -= dk; stepCount -= 1.0;

       // Take the derivatives back to single-pixel stride
       float invStride = 1.0 / stride;
       dP *= invStride; dQ.z *= invStride; dk *= invStride;

       // For this test, we don't bother checking thickness or passing the end, since we KNOW there will
       // be a hit point. As soon as
       // the ray passes behind an object, call it a hit. Advance (stride + 1) steps to fully check this
       // interval (we could skip the very first iteration, but then we'd need identical code to prime the loop)
       float refinementStepCount = 0;

       // This is the current sample point's z-value, taken back to camera space
       prevZMaxEstimate = Q.z / k;
       rayZMin = prevZMaxEstimate;

       // Ensure that the FOR-loop test passes on the first iteration since we
       // won't have a valid value of sceneZ to test.
       sceneZ = rayZMin - 1e7;

       for (;
           (refinementStepCount <= stride*1.4) &&
           (rayZMin > sceneZ) && (sceneZ != 0.0);
           P += dP, Q.z += dQ.z, k += dk, refinementStepCount += 1.0) {

           rayZMin = prevZMaxEstimate;

           // Compute the ray camera-space Z value at 1/2 fine step (pixel) into the future
           rayZMax = (dQ.z * 0.5 + Q.z) / (dk * 0.5 + k);
           rayZMax = clamp(rayZMax, zMin, zMax);

           prevZMaxEstimate = rayZMax;
           rayZMin = min(rayZMax, rayZMin);

           hitPixel = permute ? P.yx : P;
           sceneZ = texelFetch(csZBuffer, ivec2(hitPixel), 0).r;
       }

       // Undo the last increment, which happened after the test variables were set up
       Q.z -= dQ.z; refinementStepCount -= 1;

       // Count the refinement steps as fractions of the original stride. Save a register
       // by not retaining invStride until here
       stepCount += refinementStepCount / stride;
   } // refinement

    Q.xy += dQ.xy * stepCount;
	csHitPoint = Q * (1.0 / k);

    if ((P.x * stepDirection) > end) {
        // Hit the max ray distance -> blue
        user_info = vec3(0,0,1);
    } else if (stepCount >= maxSteps) {
        // Ran out of steps -> red
        user_info = vec3(1,0,0);
    } else if (sceneZ == 0.0) {
        // Went off screen -> yellow
        user_info = vec3(1,1,0);
    } else {
        // Encountered a valid hit -> green
        // ((rayZMax >= sceneZ - csZThickness) && (rayZMin <= sceneZ))
        user_info = vec3(0,1,0);
    }

    return hit;
}

float linearize_depth(float depth)
{
    const float NEAR = 0.1; // Projection matrix's near plane distance
    const float FAR = 4100.0; // Projection matrix's far plane distance

    float z = depth * 2.0 - 1.0;
    return (2.0 * NEAR * FAR) / (FAR + NEAR - z * (FAR - NEAR));
}

void main()
{
    gl_FragDepth = log_z_to_frag_depth(log_z);

	vec3 normal = normalize(normal_out);
    vec3 unscaled_normal = normalize(normal / vec3(0.22, 1.0, 0.22));
    vec3 view_normal = normalize(view_normal_out);

	vec3 camera_to_fragment = world_position_out - camera_position;

    vec3 eye = normalize(-camera_to_fragment);
    vec3 refl = reflect(-eye, normal);
	refl.y = abs(refl.y);
    vec3 reflected_color = sky_color(refl);

    float fres = fresnel(0.02, max(0, abs(dot(eye, normal))));

	vec3 spec = vec3(0);
	{
		float shininess = 35.0;

        vec3 half_vec = normalize(eye - light_direction);
        float spec_term = max(0.0, abs(dot(unscaled_normal, half_vec)));
		spec = light_color * pow(spec_term, shininess) * 30;
	}

    reflected_color += spec;

    camera_to_fragment = world_position_out - camera_position;
    float camera_to_fragment_length = length(camera_to_fragment);

#if ENABLE_REFLECTION == 1
    vec3 ray_origin = view_position_out;

    eye = normalize(-camera_to_fragment);
    refl = reflect(-eye, normal);
    refl.y = abs(refl.y);
    refl.y = max(0.05, refl.y);
    refl = normalize(refl);

    vec3 ray_direction = mat3(mat_view) * refl;
    // Scale where we start the ray according to distance from the camera
    float rayBump = max(-0.01 * ray_origin.z, 0.001);
    float z_thickness = 0.5;
    float nearz = -0.1;

    // Scale stride based on camera height:
    // Large strides break down at shallow viewing angles, so lerp the
    // stride based on a maximum height cutoff.
    float height_factor = abs(camera_position.y - world_position_out.y);
    const float full_scale_height_cutoff = 4;
    const float min_stride = 1;
    const float max_stride = 30;
    float stride = mix(min_stride, max_stride, clamp(height_factor / full_scale_height_cutoff, 0.0, 1.0));

    float jitter_fraction = stride > 1.0 ? Noise(vec2(1000, 2000) - gl_FragCoord.xy / 1280, rand_offset.x) : 0.0;
    float max_trace_steps = 12;
    float max_trace_dist = 1000;
    vec2 hit_pixel_coords;
    vec3 view_hit_point;
    vec3 user_info; // For debugging
    if(traceScreenSpaceRay1(ray_origin + ray_direction * rayBump, ray_direction, mat_proj_to_pixel, texture_depth, screen_size, z_thickness, nearz, stride, jitter_fraction, max_trace_steps, max_trace_dist, hit_pixel_coords, view_hit_point, user_info))
    {
        vec3 sky_color = reflected_color;
        reflected_color = texelFetch(texture_color, ivec2(hit_pixel_coords), 0).rgb;

        float full_fade_distance = min(max_trace_dist, stride * max_trace_steps);
        float fade_factor = length(hit_pixel_coords - gl_FragCoord.xy) / full_fade_distance;
        fade_factor *= fade_factor * fade_factor;
        float mix_factor = clamp(fade_factor, 0.0, 1.0);
        reflected_color = mix(reflected_color, sky_color, mix_factor);
    }
#endif

    vec2 water_offset = unscaled_normal.xz * 150;
    water_offset *= 20.0 / max(15.0, abs(view_position_out.z));
    water_offset = clamp(water_offset, vec2(0), screen_size - 1);
    water_offset = gl_FragCoord.xy + water_offset;
    vec3 refracted_color = texelFetch(texture_color, ivec2(water_offset), 0).rgb;

    float water_depth = texelFetch(texture_depth, ivec2(water_offset), 0).r;

    if(water_depth > view_position_out.z)
    {
        refracted_color = underwater_color;
    }

	fres = pow(fres, 2.2) * 0.3;

    vec3 color = vec3(0.0);
 
    if(is_underwater == 1)
    {
        // Scale up specular so we can see the surface better when underwater
        refracted_color.rgb += spec * 2.8;
 	
        const float water_fog_offset = 50.0;
        float water_depth_factor = clamp((1 / camera_to_fragment_length), 0, 1);

        color = mix(underwater_color, mix(refracted_color.rgb, reflected_color.rgb, fres), water_depth_factor);
    }
    else
    {
        vec3 fog_color = sky_color(camera_to_fragment / camera_to_fragment_length);
        float water_depth_factor = clamp(1.0 / exp(camera_to_fragment_length * fog_density), 0.0, 1.0);

        color = mix(fog_color, mix(refracted_color.rgb, reflected_color.rgb, fres) / 2, water_depth_factor);
    }

    //Ash
    {
        float depth_factor = 1.0 - (1.0 / exp(camera_to_fragment_length));
        color = mix(color, ash_color, depth_factor * ash_density);
    }

    color_out = vec4(color, 1.0);
    
    // float height_scale = get_height_scale_from_world(texture_wave_height_scale, world_position_out, world_offset, world_to_texture_scale);
    // color_out = vec4(vec3(pow(height_scale, 4.0), 0.0, 0.0), 1.0); 
}
