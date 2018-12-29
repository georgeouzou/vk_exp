#ifndef ORBIT_CAMERA_H
#define ORBIT_CAMERA_H

#include <glm/glm.hpp>

struct MouseState
{
	bool left;
	bool right;
	bool middle;
};

class OrbitCamera
{
private:
	glm::vec3 m_pos{ 0.0f, 0.0f, 0.0f };
	glm::vec3 m_target{ 0.0f, 0.0f, 0.0f };
	glm::vec3 m_up{ 0.0f, 1.0f, 0.0f };

	glm::vec2 m_mouse{ 0.0f, 0.0f };
	float m_wheel_offset = 0.0f;

	glm::mat4 m_view_mat{ glm::mat4(1.0f) };
	int m_width = 800;
	int m_height = 600;


public:

	OrbitCamera();

	enum Action { None, Orbit, Pan, Zoom };

	Action mouse_move(int x, int y, const MouseState &ms);
	Action mouse_scroll(float offset);

	void set_mouse_position(int x, int y);
	void set_look_at(const glm::vec3 &eye_pos,
		const glm::vec3 &target, const glm::vec3 &up);
	void set_window_size(int width, int height);

	const glm::mat4 &get_view_matrix() const;

private:
	void pan(float dx, float dy);
	void orbit(float dx, float dy);
	void zoom(float dx, float dy);
	void update();

};

#endif