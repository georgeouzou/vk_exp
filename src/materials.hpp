#ifndef MATERIALS_HPP
#define MATERIALS_HPP

// conversions are based on:
// FBX phong to PBR: https://docs.microsoft.com/en-us/azure/remote-rendering/reference/material-mapping#fbx
// FBX phong specular exponent range: https ://github.com/assimp/assimp/issues/968 
// MTL: http://paulbourke.net/dataformats/mtl/

#include <algorithm>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace materials
{

struct MTLMaterial
{
	glm::vec3 diffuse_color;
	glm::vec3 specular_color;
	float ns{ 0.0f };
};

struct PBRMaterial 
{
	glm::vec4 albedo;
	float metallic{ 0.0f };
	float roughness{ 0.0f };
	float ior{ 1.0 };
	float pad1;
};

enum class MaterialType : int32_t
{
	LAMBERTIAN = 0,
	METAL = 1,
	EMISSIVE = 2
};

static inline float luminance(glm::vec3 color) 
{
	return color.r * 0.2125 + color.g * 0.7154 + color.b * 0.0721;
}

static inline PBRMaterial convert_mtl_to_pbr(const MTLMaterial &mtl)
{
	float shininess_exponent = mtl.ns / 1000.0f; // to (0,1)
	shininess_exponent *= 100.0f; // to fbx (0,100)

	const glm::vec3 diffuse = mtl.diffuse_color;
	const glm::vec3 specular = mtl.specular_color;
	const float specular_intensity = luminance(specular);
	const float diffuse_brightness =
		0.299 * diffuse.r * diffuse.r +
		0.587 * diffuse.g * diffuse.g +
		0.114 * diffuse.b * diffuse.b;

	const float specular_brightness = 
		0.299 * specular.r * specular.r +
		0.587 * specular.g * specular.g +
		0.114 * specular.b * specular.b;

	const float specular_strength = std::max({ specular.r, specular.g, specular.b });


	PBRMaterial pbr;
	// roughness
	pbr.roughness = std::sqrt(2.0 / (shininess_exponent * specular_intensity + 2.0));
	
	// metalness
	{
		const float dsr = 0.04f; // dielectric_specular_reflectance
		const float one_minus_ss = 1.0f - specular_strength;
		const float A = dsr;
		const float B = (diffuse_brightness * (one_minus_ss / (1 - A)) + specular_brightness) - 2 * A;
		const float  C = A - specular_brightness;
		const float sq = std::sqrt(std::max(0.0f, B * B - 4 * A * C));
		const float value = (-B + sq) / (2.0f * A);
		pbr.metallic = glm::clamp(value, 0.0f, 1.0f);
	}
	
	// albedo
	{
		const float dsr = 0.04f; // dielectric_specular_reflectance
		const float one_minus_ss = 1.0f - specular_strength;
		const glm::vec3 dielectric_color = diffuse * (one_minus_ss / (1.0f - dsr) / std::max(1e-4f, 1.0f - pbr.metallic));
		const glm::vec3 metal_color = (specular - dsr * (1.0f - pbr.metallic)) * (1.0f / std::max(1e-4f, pbr.metallic));
		const glm::vec3 albedo_raw = glm::mix(dielectric_color, metal_color, pbr.metallic * pbr.metallic);
		pbr.albedo = glm::vec4(glm::clamp(albedo_raw, glm::vec3(0.0f), glm::vec3(1.0f)), 1.0f);
	}
	return pbr;
}

}




#endif