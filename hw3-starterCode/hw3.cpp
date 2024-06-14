/* **************************
 * CSCI 420
 * Assignment 3 Raytracer
 * Name: Akanksha Shukla
 * *************************
 */

#ifdef WIN32
#include <windows.h>
#endif

#if defined(WIN32) || defined(linux)
#include <GL/gl.h>
#include <GL/glut.h>
#elif defined(__APPLE__)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#define strcasecmp _stricmp
#endif

#include <imageIO.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <algorithm>
#include <iostream>
#include <cfloat>

#define MAX_TRIANGLES 20000
#define MAX_SPHERES 100
#define MAX_LIGHTS 100

char *filename = NULL;

// The different display modes.
#define MODE_DISPLAY 1
#define MODE_JPEG 2

int mode = MODE_DISPLAY;

// While solving the homework, it is useful to make the below values smaller for debugging purposes.
// The still images that you need to submit with the homework should be at the below resolution (640x480).
// However, for your own purposes, after you have solved the homework, you can increase those values to obtain higher-resolution images.
#define WIDTH 640
#define HEIGHT 480

// The field of view of the camera, in degrees.
#define fov 60.0

// Buffer to store the image when saving it to a JPEG.
unsigned char buffer[HEIGHT][WIDTH][3];

struct Vertex
{
  double position[3];
  double color_diffuse[3];
  double color_specular[3];
  double normal[3];
  double shininess;
};

struct Triangle
{
  Vertex v[3];
};

struct Sphere
{
  double position[3];
  double color_diffuse[3];
  double color_specular[3];
  double shininess;
  double radius;
};

struct Light
{
  double position[3];
  double color[3];
};

struct Color
{
  double r;
  double g;
  double b;
};

// Created a struct to store details regarding an intersection
struct Intersection
{
  bool hit;
  glm::vec3 point;
  glm::vec3 normal;
  glm::vec3 color_diffuse;
  glm::vec3 color_specular;
  double shininess;
  double reflectivity;
};

Triangle triangles[MAX_TRIANGLES];
Sphere spheres[MAX_SPHERES];
Light lights[MAX_LIGHTS];
double ambient_light[3];

int num_triangles = 0;
int num_spheres = 0;
int num_lights = 0;

void plot_pixel_display(int x, int y, unsigned char r, unsigned char g, unsigned char b);
void plot_pixel_jpeg(int x, int y, unsigned char r, unsigned char g, unsigned char b);
void plot_pixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);

const float GLOBAL_REFLECTIVITY = 0.125f;
const Color BACKGROUND_COLOR = {255.0, 255.0, 255.0};

glm::vec3 generate_ray(int x, int y)
{
  // Finds the normalized coordinates given the x, y values
  double normX = static_cast<double>(x) / (WIDTH - 1);
  double normY = static_cast<double>(y) / (HEIGHT - 1);

  // Converts the normalized coordinates into normalized device coordinates
  double ndcX = (2.0 * normX) - 1.0;
  double ndcY = (2.0 * normY) - 1.0;

  // Creates variables to store the aspect ratio and the field of view
  double aspectRatio = static_cast<double>(WIDTH) / HEIGHT;
  double fovRadians = glm::radians(fov);

  // Finds the positions on view plane that the ray passes through
  double rx = ndcX * aspectRatio * tan(fovRadians / 2.0);
  double ry = ndcY * tan(fovRadians / 2.0);

  // Creates a vector to store the normalized ray direction
  glm::vec3 ray(rx, ry, -1.0);
  ray = glm::normalize(ray);

  // Returns ray vector
  return ray;
}

bool ray_intersect_sphere(Sphere sphere, glm::vec3 ray, glm::vec3 origin, glm::vec3 &intersectionPoint)
{

  // Store coordinates of the sphere's center
  double xc = sphere.position[0];
  double yc = sphere.position[1];
  double zc = sphere.position[2];

  // Store coordinates of the ray's origin
  double x0 = origin.x;
  double y0 = origin.y;
  double z0 = origin.z;

  // Store parameters of the ray's direction vector
  double xd = ray.x;
  double yd = ray.y;
  double zd = ray.z;

  // Calculate b and c using formulas in Lecture 16, Slide 6 (a will always equal 1)
  double b = 2 * (xd * (x0 - xc) + yd * (y0 - yc) + zd * (z0 - zc));
  double c = pow((x0 - xc), 2) + pow((y0 - yc), 2) + pow((z0 - zc), 2) - pow(sphere.radius, 2);

  // Calculate the discriminant using b, c values
  double discriminant = pow(b, 2) - (4 * c);

  // Consider whether the discriminant is negative
  if (discriminant < 0.0001)
  {
    // No intersection could have occurrred
    return false;
  }

  // Find the two possible points of intersection
  double t0 = (-b + sqrt(discriminant)) / 2.0;
  double t1 = (-b - sqrt(discriminant)) / 2.0;
  // Create a double to store the intersection closest to the origin of the ray
  double t;

  // Considers if the smaller multiplier is non negative
  if (t1 > 0.0001)
  {
    // Sets the intersection to t1
    t = t1;
  }
  // Considers if the larger multiplier is non negative
  else if (t0 > 0.0001)
  {
    // Sets the intersection to t0
    t = t0;
  }
  else
  {
    // No intersection has occurred if both multipliers are negative
    return false;
  }

  // Sets the reference variable of the intersection point using the found multiplier
  intersectionPoint = origin + static_cast<float>(t) * ray;
  // Returns true since an intersection has occurred
  return true;
}

bool ray_intersect_triangle(const Triangle &triangle, const glm::vec3 &ray, const glm::vec3 &origin, glm::vec3 &intersectionPoint, glm::vec3 &intersectionNormal, glm::vec3 &baryCoords)
{
  // Store vertex positions in glm vectors for easy calculations
  glm::vec3 vertexA(triangle.v[0].position[0], triangle.v[0].position[1], triangle.v[0].position[2]);
  glm::vec3 vertexB(triangle.v[1].position[0], triangle.v[1].position[1], triangle.v[1].position[2]);
  glm::vec3 vertexC(triangle.v[2].position[0], triangle.v[2].position[1], triangle.v[2].position[2]);

  // Calculate triangle normal
  glm::vec3 normal = glm::normalize(glm::cross(vertexB - vertexA, vertexC - vertexA));

  // Calculate intersection of ray with the plane of the triangle to verify the ray isn't parallel
  float ndotRayDirection = glm::dot(normal, ray);
  if (fabs(ndotRayDirection) < 0.00001f)
  {
    // The ray is parallel to the triangle
    return false;
  }

  // Sub in ray equation into plane equation in order to find precise point of intersection
  float d = glm::dot(normal, vertexA);
  float t = (d - glm::dot(normal, origin)) / ndotRayDirection;

  // Verify that the triangle is not behind the ray's origin
  if (t < 0)
  {
    return false;
  }

  // Use the t scalar factor to find exact point of intersection on plane
  glm::vec3 P = origin + t * ray;

  // Calculate cross product in order to calculate the area of the sub-triangles
  // Dot with the normal in order to ensure consistency in area direction for barycentric coordinate
  float areaABC = glm::dot(normal, glm::cross(vertexB - vertexA, vertexC - vertexA));
  float areaPBC = glm::dot(normal, glm::cross(vertexB - P, vertexC - P));
  float areaPCA = glm::dot(normal, glm::cross(vertexC - P, vertexA - P));

  // Calculate barycentric coordinates
  float alpha = areaPBC / areaABC;
  float beta = areaPCA / areaABC;
  // Calculate gamma based on alpha and beta in order to verify that the sum of all equals 1
  float gamma = 1.0f - alpha - beta;

  // Check if point P is inside the triangle based on the 'point in triangle' test
  if (alpha >= 0.0f && beta >= 0.0f && gamma >= 0.0f)
  {
    // Set reference variable to the intersection point
    intersectionPoint = P;

    // Interpolate the normal using barycentric coordinates
    glm::vec3 normalA(triangle.v[0].normal[0], triangle.v[0].normal[1], triangle.v[0].normal[2]);
    glm::vec3 normalB(triangle.v[1].normal[0], triangle.v[1].normal[1], triangle.v[1].normal[2]);
    glm::vec3 normalC(triangle.v[2].normal[0], triangle.v[2].normal[1], triangle.v[2].normal[2]);

    // Normalize the interpolated normal
    intersectionNormal = glm::normalize(alpha * normalA + beta * normalB + gamma * normalC);

    // Set the reference barycentric coordinate variable
    baryCoords = glm::vec3(alpha, beta, gamma);

    // Return true since an intersection occurred
    return true;
  }

  // Return false, no intersection inside triangle
  return false;
}

Intersection intersect(const glm::vec3 ray, glm::vec3 origin, glm::vec3 intersectionPoint)
{
  // Create empty variables to be modified by loop
  Intersection current;
  current.hit = false;
  double minDist = std::numeric_limits<double>::infinity();

  // Loop through all spheres in the scene
  for (int i = 0; i < num_spheres; i++)
  {
    // Create an empty vector to store found intersection points
    glm::vec3 intersectionPoint;

    // Consider whether the ray stemming from origin intersects with current sphere
    if (ray_intersect_sphere(spheres[i], ray, origin, intersectionPoint))
    {

      // Compute the distance between the origin of the ray and the intersection point
      double dist = glm::length(intersectionPoint - origin);

      // Consider whether the current intersection is closer than the closest found intersection
      if (dist < minDist)
      {
        // Update the minimum distance of intersection to the current distance
        minDist = dist;
        // Update the intersection point within intersection struct
        current.point = intersectionPoint;
        // Update the boolean to indicate an intersection has occurred
        current.hit = true;

        // Store diffuse, specular and shininess values of the current sphere that has been intersected
        current.color_diffuse = glm::vec3(spheres[i].color_diffuse[0], spheres[i].color_diffuse[1], spheres[i].color_diffuse[2]);
        current.color_specular = glm::vec3(spheres[i].color_specular[0], spheres[i].color_specular[1], spheres[i].color_specular[2]);
        current.shininess = spheres[i].shininess;

        // Stores the sphere's center in a vector, and determines the distance of the ray's origin from the center
        glm::vec3 sphereCenter = glm::vec3(spheres[i].position[0], spheres[i].position[1], spheres[i].position[2]);
        double rayDistFromCenter = glm::length(origin - sphereCenter);

        // Calculates the normal at the intersection point, inverting it if the ray begins inside the sphere
        glm::vec3 normal = current.point - sphereCenter;
        normal = normal / static_cast<float>(spheres[i].radius);

        // Consider whether the ray originates inside the sphere
        if (rayDistFromCenter < spheres[i].radius)
        {
          // Invert normal and store in struct
          current.normal = -normal;
        }
        else
        {
          // Store calculated normal in intersection struct
          current.normal = normal;
        }
      }
    }
  }

  // Loop through all triangles in the scene
  for (int i = 0; i < num_triangles; i++)
  {
    // Create variables to store returns from triangle intersection function
    glm::vec3 tempPoint;
    glm::vec3 tempNormal;
    glm::vec3 baryCoords;

    if (ray_intersect_triangle(triangles[i], ray, origin, tempPoint, tempNormal, baryCoords))
    {
      // Compute the distance between the origin of the ray and the intersection point
      double dist = glm::length(tempPoint - origin);

      // Consider whether the current intersection is closer than the closest found intersection
      if (dist < minDist)
      {
        // Update the minimum distance of intersection to the current distance
        minDist = dist;
        // Update the intersection point within intersection struct
        current.point = tempPoint;
        // Store returned (interpolated) normal in intersection struct
        current.normal = tempNormal;
        // Update the boolean to indicate an intersection has occurred
        current.hit = true;

        // Interpolate the diffuse property of the pixel based on barycentric coordinates
        current.color_diffuse = baryCoords.x * glm::vec3(triangles[i].v[0].color_diffuse[0], triangles[i].v[0].color_diffuse[1], triangles[i].v[0].color_diffuse[2]) +
                                baryCoords.y * glm::vec3(triangles[i].v[1].color_diffuse[0], triangles[i].v[1].color_diffuse[1], triangles[i].v[1].color_diffuse[2]) +
                                baryCoords.z * glm::vec3(triangles[i].v[2].color_diffuse[0], triangles[i].v[2].color_diffuse[1], triangles[i].v[2].color_diffuse[2]);

        // Interpolate the specular property of the pixel based on barycentric coordinates
        current.color_specular = baryCoords.x * glm::vec3(triangles[i].v[0].color_specular[0], triangles[i].v[0].color_specular[1], triangles[i].v[0].color_specular[2]) +
                                 baryCoords.y * glm::vec3(triangles[i].v[1].color_specular[0], triangles[i].v[1].color_specular[1], triangles[i].v[1].color_specular[2]) +
                                 baryCoords.z * glm::vec3(triangles[i].v[2].color_specular[0], triangles[i].v[2].color_specular[1], triangles[i].v[2].color_specular[2]);

        // Interpolate the shininess property of the pixel based on barycentric coordinates
        current.shininess = baryCoords.x * triangles[i].v[0].shininess +
                            baryCoords.y * triangles[i].v[1].shininess +
                            baryCoords.z * triangles[i].v[2].shininess;
      }
    }
  }

  // Regardless of shape, set the reflectivity of the material to the global reflectivity
  current.reflectivity = GLOBAL_REFLECTIVITY;

  // Return the populated intersection struct of the closest intersection
  return current;
}

float shadow_test(glm::vec3 intersectionPoint, int lightIndex)
{
  // Store current light's position in a glm::vec3 for easy calculations
  glm::vec3 lightPos(lights[lightIndex].position[0], lights[lightIndex].position[1], lights[lightIndex].position[2]);
  // Define radius for area light centered at above position
  double radius = 0.25;
  // Count to track the number of rays that are not obstructed
  int successfulRays = 0;

  // Generate 16 samples for each light
  for (int i = 0; i < 16; i++)
  {

    // Calculate a random angle between 0 to 2pi
    double angle = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
    // Calculate a random distance between 0 and the specified radius of the area light
    double distance = static_cast<double>(rand()) / RAND_MAX * radius;

    // Uses the angle and distance to find cartesian coordinates offset from the area light's center
    glm::vec3 randomPos = lightPos + glm::vec3(cos(angle) * distance, sin(angle) * distance, 0);

    // Find the vector from the current pixel to the current light
    glm::vec3 lightDirection = randomPos - intersectionPoint;
    // Compute the distance of the light from the pixel
    double lightDistance = glm::length(lightDirection);
    // Normalize the direction vector from the pixel to the light
    lightDirection = glm::normalize(lightDirection);

    // Offset shadow origin slightly towards the light to avoid self-shadowing
    glm::vec3 shadowOrigin = intersectionPoint + lightDirection * 0.01f;

    // Reuse intersect function to find any obstruction between point and light
    glm::vec3 shadowIntersect;
    Intersection shadow = intersect(lightDirection, shadowOrigin, shadowIntersect);
    // Considers if there was an obstruction between the pixel and the light
    if (shadow.hit)
    {
      // Calculate the distance between the current pixel and the obstruction
      double obstructionDistance = glm::length(shadow.point - shadowOrigin);
      // Check if the obstruction is actually between the point and the light
      if (obstructionDistance < lightDistance)
      {
        // There is a shadow cast on the current pixel
        continue;
      }
    }

    // There is no shadow cast by this ray, it is a successful ray
    successfulRays++;
  }

  // Return the proportion of successful rays from the total amount of rays
  return static_cast<double>(successfulRays) / 16;
}

Color phong_shading(const Intersection &obj, int lightIndex)
{

  // Cast the shadow test in order to determine the intensity of light on the pixel
  float lightIntensity = shadow_test(obj.point, lightIndex);

  // Store the light properties in vectors for ease of calculation (multiply the color by the light intensity found by the shadow test)
  glm::vec3 lightColor = glm::vec3(lights[lightIndex].color[0], lights[lightIndex].color[1], lights[lightIndex].color[2]) * lightIntensity;
  glm::vec3 lightPosition = glm::vec3(lights[lightIndex].position[0], lights[lightIndex].position[1], lights[lightIndex].position[2]);

  // Create the l unit vector by normalizing the vector from the current pixel to the light
  glm::vec3 toLight = glm::normalize(lightPosition - obj.point);
  // Create the n unit vector by using the current pixel's normal
  glm::vec3 normal = glm::normalize(obj.normal);
  // Create the v unit vector by normalizing the vector from the current pixel to the camera
  glm::vec3 toViewer = glm::normalize(glm::vec3(0, 0, 0) - obj.point);
  // Create the r unit vector using the formula from the slides
  glm::vec3 reflection = glm::normalize(2 * glm::dot(toLight, normal) * normal - toLight);

  // Calculate the diffuse component, clamping at 0 if the dot product is negative
  double diffVectors = std::max(glm::dot(normal, toLight), 0.0f);
  glm::vec3 diffuse = obj.color_diffuse * static_cast<float>(diffVectors);

  // Calculate the specular component, clamping at 0 if the dot product is negative
  double specVectors = std::max(glm::dot(reflection, toViewer), 0.0f);
  double specularCoefficient = pow(specVectors, obj.shininess);
  glm::vec3 specular = obj.color_specular * static_cast<float>(specularCoefficient);

  // Find the total light contribution using formula from lecture
  glm::vec3 color = lightColor * (diffuse + specular);

  // Ensure that the color component remains in the [0,1] range
  color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));

  // Convert the color to be in char format for plotting
  Color outputColor = {color.r * 255, color.g * 255, color.b * 255};
  return outputColor;
}

Color recursive_ray_trace(glm::vec3 ray, glm::vec3 origin, int depth)
{

  // Only recurse up to a depth of 3, else return a black color component
  if (depth >= 3)
  {
    Color black = {0, 0, 0};
    return black;
  }

  // Create variable to be populated by helper functions
  glm::vec3 intersectionPoint;
  // Call the intersect point on the given ray at the current pixel
  Intersection current = intersect(ray, origin, intersectionPoint);

  // Compute the color the ray contributes (based on intersections and lighting)
  if (current.hit)
  {
    Color lightContributions = {0, 0, 0};

    // Loop through all the lights in the scene
    for (int i = 0; i < num_lights; i++)
    {
      // Calculate the light contribution via phong shading
      Color contribution = phong_shading(current, i);
      lightContributions.r += contribution.r;
      lightContributions.g += contribution.g;
      lightContributions.b += contribution.b;
    }

    // Add ambient light influences
    lightContributions.r += ambient_light[0] * 255;
    lightContributions.g += ambient_light[1] * 255;
    lightContributions.b += ambient_light[2] * 255;

    // Calculate the reflected ray vector
    glm::vec3 reflectedRay = ray - 2 * glm::dot(ray, current.normal) * current.normal;
    // Calculate the origin of the reflected ray (with a small offset to avoid self-intersection)
    glm::vec3 reflectedOrigin = current.point + reflectedRay * 0.01f;
    // Call the same function again in order to compute the colour from the reflection
    Color reflectedColour = recursive_ray_trace(reflectedRay, reflectedOrigin, depth + 1);

    // Blend reflected color with local contributions based on reflectivity
    lightContributions.r = (1 - current.reflectivity) * lightContributions.r + current.reflectivity * reflectedColour.r;
    lightContributions.g = (1 - current.reflectivity) * lightContributions.g + current.reflectivity * reflectedColour.g;
    lightContributions.b = (1 - current.reflectivity) * lightContributions.b + current.reflectivity * reflectedColour.b;

    return lightContributions;
  }
  else
  {
    // Background color contribution (assuming white background)
    return BACKGROUND_COLOR;
  }
}

void ray_trace(int x, int y)
{
  // Define a 4x4 supersampling grid
  int num_samples_per_axis = 4;
  // Calculate the sample weight of each iteration of the for loop to ensure the average color is found
  float sample_weight = 1.0 / (num_samples_per_axis * num_samples_per_axis);
  Color pixelColor = {0, 0, 0};

  // Loop through the samples in the 4x4 grid
  for (int sx = 0; sx < num_samples_per_axis; ++sx)
  {
    for (int sy = 0; sy < num_samples_per_axis; ++sy)
    {

      // Calculate sub-pixel offsets, ensuring that we are targeting the center of each subpixel
      float sub_x = (sx + 0.5f) * (1.0f / num_samples_per_axis);
      float sub_y = (sy + 0.5f) * (1.0f / num_samples_per_axis);

      // Generate a ray that passes through this sub-pixel
      glm::vec3 ray = generate_ray(x + sub_x - 0.5f, y + sub_y - 0.5f);
      // Call the recursive ray trace function in order to capture the lighting from reflections
      Color sampleColor = recursive_ray_trace(ray, glm::vec3(0, 0, 0), 0);

      // Accumulate contributions for the pixel, dividing by weight to ensure the output is averaged
      pixelColor.r += sampleColor.r * sample_weight;
      pixelColor.g += sampleColor.g * sample_weight;
      pixelColor.b += sampleColor.b * sample_weight;
    }
  }

  // Clamp the colors at 255.0 to ensure accurate plotting
  pixelColor.r = std::min(255.0, pixelColor.r);
  pixelColor.g = std::min(255.0, pixelColor.g);
  pixelColor.b = std::min(255.0, pixelColor.b);

  // Plot the pixel with the accumulated color
  plot_pixel(x, y, static_cast<unsigned char>(pixelColor.r), static_cast<unsigned char>(pixelColor.g), static_cast<unsigned char>(pixelColor.b));
}

void draw_scene()
{
  for (unsigned int x = 0; x < WIDTH; x++)
  {
    glPointSize(2.0);
    glBegin(GL_POINTS);
    for (unsigned int y = 0; y < HEIGHT; y++)
    {
      // Calls the ray trace function for the current coordinates
      ray_trace(x, y);
    }
    glEnd();
    glFlush();
  }
  printf("Ray tracing completed.\n");
  fflush(stdout);
}

void plot_pixel_display(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
  glColor3f(((float)r) / 255.0f, ((float)g) / 255.0f, ((float)b) / 255.0f);
  glVertex2i(x, y);
}

void plot_pixel_jpeg(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
  buffer[y][x][0] = r;
  buffer[y][x][1] = g;
  buffer[y][x][2] = b;
}

void plot_pixel(int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
  plot_pixel_display(x, y, r, g, b);
  if (mode == MODE_JPEG)
    plot_pixel_jpeg(x, y, r, g, b);
}

void save_jpg()
{
  printf("Saving JPEG file: %s\n", filename);

  ImageIO img(WIDTH, HEIGHT, 3, &buffer[0][0][0]);
  if (img.save(filename, ImageIO::FORMAT_JPEG) != ImageIO::OK)
    printf("Error in saving\n");
  else
    printf("File saved successfully\n");
}

void parse_check(const char *expected, char *found)
{
  if (strcasecmp(expected, found))
  {
    printf("Expected '%s ' found '%s '\n", expected, found);
    printf("Parsing error; abnormal program abortion.\n");
    exit(0);
  }
}

void parse_doubles(FILE *file, const char *check, double p[3])
{
  char str[100];
  fscanf(file, "%s", str);
  parse_check(check, str);
  fscanf(file, "%lf %lf %lf", &p[0], &p[1], &p[2]);
  printf("%s %lf %lf %lf\n", check, p[0], p[1], p[2]);
}

void parse_rad(FILE *file, double *r)
{
  char str[100];
  fscanf(file, "%s", str);
  parse_check("rad:", str);
  fscanf(file, "%lf", r);
  printf("rad: %f\n", *r);
}

void parse_shi(FILE *file, double *shi)
{
  char s[100];
  fscanf(file, "%s", s);
  parse_check("shi:", s);
  fscanf(file, "%lf", shi);
  printf("shi: %f\n", *shi);
}

int loadScene(char *argv)
{
  FILE *file = fopen(argv, "r");
  if (!file)
  {
    printf("Unable to open input file %s. Program exiting.\n", argv);
    exit(0);
  }

  int number_of_objects;
  char type[50];
  Triangle t;
  Sphere s;
  Light l;
  fscanf(file, "%i", &number_of_objects);

  printf("number of objects: %i\n", number_of_objects);

  parse_doubles(file, "amb:", ambient_light);

  for (int i = 0; i < number_of_objects; i++)
  {
    fscanf(file, "%s\n", type);
    printf("%s\n", type);
    if (strcasecmp(type, "triangle") == 0)
    {
      printf("found triangle\n");
      for (int j = 0; j < 3; j++)
      {
        parse_doubles(file, "pos:", t.v[j].position);
        parse_doubles(file, "nor:", t.v[j].normal);
        parse_doubles(file, "dif:", t.v[j].color_diffuse);
        parse_doubles(file, "spe:", t.v[j].color_specular);
        parse_shi(file, &t.v[j].shininess);
      }

      if (num_triangles == MAX_TRIANGLES)
      {
        printf("too many triangles, you should increase MAX_TRIANGLES!\n");
        exit(0);
      }
      triangles[num_triangles++] = t;
    }
    else if (strcasecmp(type, "sphere") == 0)
    {
      printf("found sphere\n");

      parse_doubles(file, "pos:", s.position);
      parse_rad(file, &s.radius);
      parse_doubles(file, "dif:", s.color_diffuse);
      parse_doubles(file, "spe:", s.color_specular);
      parse_shi(file, &s.shininess);

      if (num_spheres == MAX_SPHERES)
      {
        printf("too many spheres, you should increase MAX_SPHERES!\n");
        exit(0);
      }
      spheres[num_spheres++] = s;
    }
    else if (strcasecmp(type, "light") == 0)
    {
      printf("found light\n");
      parse_doubles(file, "pos:", l.position);
      parse_doubles(file, "col:", l.color);

      if (num_lights == MAX_LIGHTS)
      {
        printf("too many lights, you should increase MAX_LIGHTS!\n");
        exit(0);
      }
      lights[num_lights++] = l;
    }
    else
    {
      printf("unknown type in scene description:\n%s\n", type);
      exit(0);
    }
  }
  return 0;
}

void display()
{
}

void init()
{
  glMatrixMode(GL_PROJECTION);
  glOrtho(0, WIDTH, 0, HEIGHT, 1, -1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
}

void idle()
{
  // Hack to make it only draw once.
  static int once = 0;
  if (!once)
  {
    draw_scene();
    if (mode == MODE_JPEG)
      save_jpg();
  }
  once = 1;
}

int main(int argc, char **argv)
{
  if ((argc < 2) || (argc > 3))
  {
    printf("Usage: %s <input scenefile> [output jpegname]\n", argv[0]);
    exit(0);
  }
  if (argc == 3)
  {
    mode = MODE_JPEG;
    filename = argv[2];
  }
  else if (argc == 2)
    mode = MODE_DISPLAY;

  glutInit(&argc, argv);
  loadScene(argv[1]);

  glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
  glutInitWindowPosition(0, 0);
  glutInitWindowSize(WIDTH, HEIGHT);
  int window = glutCreateWindow("Ray Tracer");
#ifdef __APPLE__
  // This is needed on recent Mac OS X versions to correctly display the window.
  glutReshapeWindow(WIDTH - 1, HEIGHT - 1);
#endif
  glutDisplayFunc(display);
  glutIdleFunc(idle);
  init();
  glutMainLoop();
}
