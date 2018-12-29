
#include "orbit_camera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <limits>
#include <glm/gtx/rotate_vector.hpp>

// based on Nvidia's manipulator but simpler

namespace detail {

static bool float_is_zero(float a)
{
	return fabs(a) < std::numeric_limits<float>::epsilon();
}
static float sign(float s)
{
	return (s < 0.f) ? -1.f : 1.f;
}
static double sign(double s)
{
	return (s < 0.0) ? -1.0 : 1.0;
}

}

OrbitCamera::OrbitCamera()
{
	update();
}

OrbitCamera::Action OrbitCamera::mouse_move(int x, int y, const MouseState &ms)
{
	Action act = None;
	if (ms.left) {
		act = Orbit;
	} else if (ms.middle || ms.right) {
		act = Pan;
	}
	if (act == None) {
		return None;
	}

	float dx = float(x - m_mouse.x) / float(m_width);
	float dy = float(y - m_mouse.y) / float(m_height);
	if (act == Orbit) {
		orbit(dx, dy);
	} else if (act == Pan) {
		pan(-dx, -dy);
	}
	
	update();

	set_mouse_position(x, y);

	return act;
}

OrbitCamera::Action OrbitCamera::mouse_scroll(float offset)
{
	float dx = (offset * fabs(offset)) / static_cast<float>(m_width);

	const float speed = 30.0f;
	zoom(dx * speed, dx * speed);
	update();

	return Zoom;
}

void OrbitCamera::set_mouse_position(int x, int y)
{
	m_mouse.x = static_cast<float>(x);
	m_mouse.y = static_cast<float>(y);
}

void OrbitCamera::set_look_at(const glm::vec3 &eye_pos,
	const glm::vec3 &target, const glm::vec3 &up)
{
	m_pos = eye_pos;
	m_target = target;
	m_up = up;
	update();
}

void OrbitCamera::set_window_size(int width, int height)
{
	m_width = width;
	m_height = height;
}

const glm::mat4 &OrbitCamera::get_view_matrix() const
{
	return m_view_mat;
}

void OrbitCamera::pan(float dx, float dy)
{
	glm::vec3 z(m_pos - m_target);
	float length = static_cast<float>(glm::length(z)) / 0.785f; // 45 degrees
	z = glm::normalize(z);
	glm::vec3 x = glm::cross(m_up, z);
	x = glm::normalize(x);
	glm::vec3 y = glm::cross(z, x);
	y = glm::normalize(y);
	x *= -dx * length;
	y *= dy * length;

	m_pos += x + y;
	m_target += x + y;
}

void OrbitCamera::orbit(float dx, float dy)
{
	if (detail::float_is_zero(dx) && detail::float_is_zero(dy))
		return;

	dx *= float(glm::two_pi<float>());
	dy *= float(glm::two_pi<float>());

	const bool invert = false;
	glm::vec3 origin(invert ? m_pos : m_target);
	glm::vec3 position(invert ? m_target : m_pos);
	glm::vec3 center_to_eye(position - origin);
	float radius = glm::length(center_to_eye);
	center_to_eye = glm::normalize(center_to_eye);

	glm::mat4 rot_x, rot_y;

	// Find the rotation around up axis
	glm::vec3 axe_z(glm::normalize(center_to_eye));
	rot_y = glm::rotate(dx, m_up);
	// Apply the y rotation to the eye-center vector
	glm::vec4 vect_tmp = rot_y * glm::vec4(center_to_eye.x, center_to_eye.y, center_to_eye.z, 0);
	center_to_eye = glm::vec3(vect_tmp.x, vect_tmp.y, vect_tmp.z);
	// Find the rotation around the x vector
	glm::vec3 axe_x = glm::cross(m_up, axe_z);
	axe_x = glm::normalize(axe_x);
	rot_x = glm::rotate(dy, axe_x);
	// Apply the x rotation to the eye-center vector
	vect_tmp = rot_x * glm::vec4(center_to_eye.x, center_to_eye.y, center_to_eye.z, 0);
	glm::vec3 vect_rot(vect_tmp.x, vect_tmp.y, vect_tmp.z);
	if (detail::sign(vect_rot.x) == detail::sign(center_to_eye.x))
		center_to_eye = vect_rot;
	// Make the vector as long as it was originally
	center_to_eye *= radius;
	// Finding the new position
	glm::vec3 newPosition = center_to_eye + origin;
	if (!invert) {
		m_pos = newPosition; 
	} else {
		m_target = newPosition;
	}
}

void OrbitCamera::zoom(float dx, float dy)
{
	glm::vec3 z = m_target - m_pos;
	float length = static_cast<float>(glm::length(z));
	
	if (detail::float_is_zero(length))
		return;

	const float speed = 30.0f;
	float dd;
	
	dd = fabs(dx) > fabs(dy) ? dx : -dy;
	float factor = speed * dd / length;

	length /= 10;
	length = length < 0.001f ? 0.001f : length;
	factor *= length;

	if (factor >= 1.0f)
		return;

	z *= factor;
	m_pos += z;
	m_target += z;
}

void OrbitCamera::update()
{
	m_view_mat = glm::lookAt(m_pos, m_target, m_up);
}

