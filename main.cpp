#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <vector>
#include <memory>
#include <map>
#include <iostream>
#include <cstdio>
#include <GL/gl.h>
// ----------------------------------------------------------------------------------
// ШЕЙДЕРЫ
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
// ШЕЙДЕРЫ (должны быть в самом начале файла!)
// ----------------------------------------------------------------------------------
static const char* vertexShader = R"glsl(
#version 330 core
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;

uniform mat4 mvp;
uniform mat4 matModel;

out vec3 fragWorldPos;
out vec2 fragTexCoord;
out vec3 fragNormal;

void main() {
    vec4 worldPos = matModel * vec4(vertexPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    fragTexCoord = vertexTexCoord;
    fragNormal   = normalize(mat3(matModel) * vertexNormal);
    gl_Position  = mvp * vec4(vertexPosition, 1.0);
}
)glsl";

static const char* fragmentShader = R"glsl(
#version 330 core
in vec3 fragWorldPos;
in vec2 fragTexCoord;
in vec3 fragNormal;

out vec4 finalColor;

uniform sampler2D texture0;
uniform vec4      colDiffuse;
uniform vec3      lightPos;
uniform vec3      viewPos;
uniform vec3      playerPos;
uniform vec3      lightColor = vec3(1.0, 0.95, 0.88);

uniform float metallic  = 0.0;
uniform float roughness = 0.85;
uniform float ao        = 1.0;

// Shadow mapping
uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float a) {
    float a2 = a * a;
    float NdotH2 = max(dot(N, H), 0.0);
    NdotH2 *= NdotH2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float k) {
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float k) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, k) * GeometrySchlickGGX(NdotL, k);
}

float ShadowCalculation(vec3 fragPos, vec3 normal, vec3 lightDir) {
    vec4 fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

        // Проверка на выход за границы shadow map
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || 
        projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 0.0;
    }

    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;

    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.001);
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);

    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += (currentDepth - bias) > pcfDepth ? 1.0 : 0.0;

        }
    }
    shadow /= 9.0;
    return shadow;
}

void main() {
    vec4 texColor = texture(texture0, fragTexCoord) * colDiffuse;
    if (texColor.a < 0.1) discard;

    vec3 albedo = texColor.rgb;
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(viewPos - fragWorldPos);
    vec3 L = normalize(lightPos - fragWorldPos);
    vec3 H = normalize(V + L);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NdotL = max(dot(N, L), 0.0);

    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, k);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (D * G * F) / max(4.0 * max(dot(N, V), 0.0) * NdotL, 0.001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    vec3 Lo = (kD * albedo / PI + specular) * lightColor * NdotL;

    vec3 ambient = vec3(0.04) * albedo * ao;

    // Тень от солнца
    float shadow = ShadowCalculation(fragWorldPos, N, L);

    vec3 lighting = ambient + Lo * (1.0 - shadow * 0.75);

    // Тонемаппинг + гамма
    vec3 color = lighting;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    finalColor = vec4(color, texColor.a);
}
)glsl";
// ... твои старые шейдеры (vertexShader, fragmentShader) ...

static const char* shadowVertexShader = R"glsl(
#version 330 core
in vec3 vertexPosition;

uniform mat4 mvp;
uniform mat4 matModel;

void main() {
    gl_Position = mvp * matModel * vec4(vertexPosition, 1.0);
}
)glsl";
static const char* shadowFragmentShader = R"glsl(
#version 330 core
void main() {
    // Здесь пусто, так как нам нужна только глубина (gl_FragDepth заполняется автоматически)
}
)glsl";
// ----------------------------------------------------------------------------------
// ПРЕДВАРИТЕЛЬНЫЕ ОБЪЯВЛЕНИЯ СТРУКТУР (ЧТОБЫ ИЗБЕЖАТЬ ОШИБОК КОМПИЛЯЦИИ)
// ----------------------------------------------------------------------------------
void Update3DSound(Sound sound, Vector3 sourcePos, Vector3 listenerPos, float maxDist) {
    float dist = Vector3Distance(sourcePos, listenerPos);
    
    // Громкость: затухает до 0 на дистанции maxDist
    float volume = 1.0f - Clamp(dist / maxDist, 0.0f, 1.0f);
    SetSoundVolume(sound, volume * volume); // Квадратичное затухание для реализма

    // Панорама (Pan): 0.0 - лево, 1.0 - право.
    // Вычисляем, где объект относительно "взгляда" игрока
    Vector3 dir = Vector3Normalize(Vector3Subtract(sourcePos, listenerPos));
    float pan = 0.5f + (dir.x * 0.4f); // Смещаем баланс в зависимости от X
    SetSoundPan(sound, Clamp(pan, 0.0f, 1.0f));
}
// Класс снаряда
struct Projectile {
    Vector3 pos;
    Vector3 vel;
    float lifeTime = 3.0f;
    bool active = true;
    bool soundStarted = false;
    void Update(float dt, Model &map, Vector3 listenerPos,Sound flySound, Sound expSound) {
        
        if (!active) return;
        if (flySound.frameCount > 0) {
            if (!soundStarted) {
                PlaySound(flySound);
                soundStarted = true;
            }
            Update3DSound(flySound, pos, listenerPos, 15.0f);
        }
        pos = Vector3Add(pos, Vector3Scale(vel, dt));
        lifeTime -= dt;
        if (lifeTime <= 0){
            active = false;
            // StopSound(flySound);
            PlaySound(expSound);
        }
        Ray ray = { pos, Vector3Normalize(vel) };
        for(int i=0; i<map.meshCount; i++) {
            RayCollision col = GetRayCollisionMesh(ray, map.meshes[i], map.transform);
            if (col.hit && col.distance < 0.5f){
                active = false;
                // StopSound(flySound);
                PlaySound(expSound);
            }
        }
    }
};

// Класс Актера (Враги и Союзники)
class Actor {
public:
    Vector3 pos;
    Vector3 vel = {0,0,0};
    float health = 100.0f;
    bool isEnemy;
    bool following = false;
    Texture2D tex;
    float height = 2.0f;
    float shootTimer = 0.0f;

    Actor(Vector3 startPos, bool enemy, const char* texPath);
    ~Actor();
    void Update(float dt, Vector3 playerPos, Model &map, std::vector<std::unique_ptr<Actor>> &allActors, std::vector<Projectile> &projectiles);
    void Draw(const Camera3D& camera);
};
void ResolveMapCollision(Vector3 &pos, Vector3 &vel, float radius, float height, Model &map) {
    // 1. Сначала гравитация и пол
    bool onGround = false;
    Vector3 downRayPos = Vector3Add(pos, {0, 0.5f, 0}); // Луч из "колен"
    Ray rayDown = { downRayPos, {0, -1, 0} };

    for (int i = 0; i < map.meshCount; i++) {
        RayCollision col = GetRayCollisionMesh(rayDown, map.meshes[i], map.transform);
        if (col.hit && col.distance < 0.51f) { // 0.5f (смещение) + запас
            pos.y = downRayPos.y - col.distance;
            if (vel.y < 0) vel.y = 0;
            onGround = true;
        }
    }

    // 2. Горизонтальные коллизии (Стены и Колонны)
    // Пускаем лучи на уровне "пояса" (height * 0.5f)
    Vector3 horizontalOrigin = Vector3Add(pos, {0, height * 0.5f, 0});
    Vector3 dirs[4] = {{1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}};

    for (int i = 0; i < map.meshCount; i++) {
        for (auto& d : dirs) {
            // Хитрость: пускаем луч извне в сторону игрока
            // или пускаем луч чуть дальше радиуса игрока
            Ray ray = { horizontalOrigin, d };
            RayCollision col = GetRayCollisionMesh(ray, map.meshes[i], map.transform);

            if (col.hit && col.distance < radius) {
                // Выталкиваем строго в противоположную сторону от стены
                float pushDist = radius - col.distance;
                pos = Vector3Subtract(pos, Vector3Scale(d, pushDist));
                
                // Обнуляем скорость в сторону стены, чтобы не "липнуть"
                if (d.x != 0) vel.x = 0;
                if (d.z != 0) vel.z = 0;
            }
        }
    }
}

// Объект, который можно таскать (Prop)
struct Prop {
    Vector3 pos;
    Vector3 vel = {0,0,0}; // Добавляем скорость
    bool isHeld = false;
    float radius = 0.5f;
    float height = 1.0f;   // Высота для коллизий
    bool isRadio = false;
    Sound music;
    void Update(Vector3 listenerPos, Vector3 forward, float dt, Model &map) {
        if (isRadio && music.frameCount > 0) { 
                if (!IsSoundPlaying(music)) PlaySound(music);
                Update3DSound(music, pos, listenerPos, 20.0f);
        }

        if (isHeld) {
            Vector3 targetPos = Vector3Add(listenerPos, Vector3Scale(forward, 2.0f));
            pos = Vector3Lerp(pos, targetPos, 15.0f * dt);
            vel = {0,0,0}; // Обнуляем скорость, пока в руках
        } else {
            vel.y -= 25.0f * dt;
            pos = Vector3Add(pos, Vector3Scale(vel, dt));

            // Применяем ту же физику, что у игрока и врагов!
            ResolveMapCollision(pos, vel, radius, height, map);
            
            // Защита от провала в бездну
            if (pos.y < -15.0f) {
                pos.y = 10.0f;
                vel = {0,0,0};
            }
        }
    }

    void Draw() {
        Vector3 visualPos = { pos.x, pos.y + 0.5f, pos.z };
        DrawCube(visualPos, 1.0f, 1.0f, 1.0f, ORANGE);
        // DrawCubeWires(pos, 1.1f, 1.1f, 1.1f, BLACK);
    }
};

// База спавна
struct SpawnBase {
    Vector3 position;
    bool isCaptured = false;

    SpawnBase(Vector3 p) : position(p) {}

    // Обновляем состояние захвата: если рядом Игрок или Союзник -> база захвачена
    void UpdateCaptureState(const Vector3& playerPos, const std::vector<std::unique_ptr<Actor>>& actors) {
        bool friendNearby = false;
        float captureRadius = 10.0f;

        if (Vector3Distance(position, playerPos) < captureRadius) friendNearby = true;
        
        for (auto& a : actors) {
            // Тег "com_a" (союзник)
            if (!a->isEnemy && a->health > 0 && Vector3Distance(position, a->pos) < captureRadius) {
                friendNearby = true;
                break;
            }
        }
        isCaptured = friendNearby;
    }
};

// ----------------------------------------------------------------------------------
// ГЛОБАЛЬНАЯ ФИЗИКА
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
// РЕАЛИЗАЦИЯ МЕТОДОВ ACTOR (нужна после объявления ResolveMapCollision)
// ----------------------------------------------------------------------------------
Actor::Actor(Vector3 startPos, bool enemy, const char* texPath) : pos(startPos), isEnemy(enemy) {
    if (FileExists(texPath)) {
        tex = LoadTexture(texPath);
        SetTextureFilter(tex, TEXTURE_FILTER_POINT);
    } else {
        Image img = GenImageColor(64, 64, enemy ? RED : GREEN);
        tex = LoadTextureFromImage(img);
        UnloadImage(img);
    }
}

Actor::~Actor() { UnloadTexture(tex); }

void Actor::Update(float dt, Vector3 playerPos, Model &map, std::vector<std::unique_ptr<Actor>> &allActors, std::vector<Projectile> &projectiles) {
    if (health <= 0) return;

    float distToPlayer = Vector3Distance(pos, playerPos);
    Vector3 moveDir = {0,0,0};

    // --- ЛОГИКА ВРАГА ---
    if (isEnemy) {
        // Враги всегда видят игрока и идут к нему (удалено условие дистанции)
        moveDir = Vector3Normalize(Vector3Subtract(playerPos, pos));
        pos = Vector3Add(pos, Vector3Scale(moveDir, 3.5f * dt));
    } 
    // --- ЛОГИКА СОЮЗНИКА ---
    else {
        if (IsKeyPressed(KEY_E) && distToPlayer < 3.0f) {
            following = !following;
        }
        if (following && distToPlayer > 4.0f) {
            moveDir = Vector3Normalize(Vector3Subtract(playerPos, pos));
            pos = Vector3Add(pos, Vector3Scale(moveDir, 4.5f * dt));
        }

        // Авто-стрельба
        shootTimer += dt;
        if (shootTimer > 1.5f) { 
            for (auto& other : allActors) {
                if (other->isEnemy && other->health > 0) {
                    float distToEnemy = Vector3Distance(pos, other->pos);
                    if (distToEnemy < 20.0f) {
                        Vector3 aimDir = Vector3Normalize(Vector3Subtract(other->pos, pos));
                        Projectile proj;
                        proj.pos = Vector3Add(pos, {0, 1.5f, 0});
                        proj.vel = Vector3Scale(aimDir, 15.0f);
                        projectiles.push_back(proj);
                        shootTimer = 0;
                        break;
                    }
                }
            }
        }
    }

    // Физика
    vel.y -= 25.0f * dt;
    pos.y += vel.y * dt;
    ResolveMapCollision(pos, vel, 0.6f, height, map);
    if (pos.y < -20.0f) { // Если упал слишком низко
        pos.y = 10.0f; 
        vel.y = 0;
        // Опционально: можно чуть-чуть сбросить горизонтальную скорость, 
        // чтобы они не вылетали как из пушки после телепортации
        vel.x = 0;
        vel.z = 0;
    }
}

void Actor::Draw(const Camera3D& camera) {
    if (tex.id == 0 || health <= 0) return;
    Color tint = isEnemy ? RED : WHITE;
    DrawBillboard(camera, tex, {pos.x, pos.y + height/2, pos.z}, height, tint);
    if (!isEnemy && following) {
        Vector3 textPos = Vector3Add(pos, {0, 2.5f, 0});
        DrawBillboard(camera, tex, textPos, 0.5f, GREEN);
    }
}

// ----------------------------------------------------------------------------------
// АНИМАЦИИ ИГРОКА
// ----------------------------------------------------------------------------------
struct SpriteAnimation {
    std::vector<Texture2D> frames;
    float frameRate = 10.0f;
    float timer = 0.0f;
    int currentFrame = 0;
    void AddFrame(const char* path) { if (FileExists(path)) { Texture2D t = LoadTexture(path); SetTextureFilter(t, TEXTURE_FILTER_POINT); frames.push_back(t); } }
    void Update(float dt) { if (frames.size() <= 1) return; timer += dt; if (timer >= 1.0f / frameRate) { timer = 0; currentFrame = (currentFrame + 1) % frames.size(); } }
    Texture2D Get() { return frames.empty() ? Texture2D{0} : frames[currentFrame]; }
};

enum PlayerState { IDLE, WALK, WALK_SIDE };

class Player {
public:
    Vector3 pos;
    Vector3 vel = {0,0,0};
    float camH = 0, camV = 0.3f;
    float radius = 0.6f, height = 2.0f;
    bool flipX = false;
    float speed = 7.0f;
    float targetHeight = 2.0f;
    Prop* grabbedProp = nullptr; 
    PlayerState state = IDLE;
    std::map<PlayerState, SpriteAnimation> anims;

    Player(Vector3 startPos) : pos(startPos) {
        anims[IDLE].AddFrame("textures/player/idle_0.png");
        for(int i=0; i<4; i++) {
            anims[WALK].AddFrame(TextFormat("textures/player/walk_%i.png", i));
            anims[WALK_SIDE].AddFrame(TextFormat("textures/player/side_walk_%i.png", i));
        }
    }

    // Важно: Принимаем std::vector<Prop>& props, чтобы видеть ящики из GameWorld
    void Update(float dt, Model &map, std::vector<Prop>& props) {
        speed = IsKeyDown(KEY_LEFT_SHIFT) ? 12.0f : 7.0f;
        if (IsKeyDown(KEY_LEFT_CONTROL)) speed *= 0.5f; 

        height = Lerp(height, targetHeight, 10.0f * dt);
        camH += GetMouseDelta().x * 0.003f;
        camV = Clamp(camV + GetMouseDelta().y * 0.003f, -1.5f, 1.5f);

        Vector3 fwd = Vector3Normalize({-sinf(camH), 0, -cosf(camH)});
        Vector3 rgt = {-fwd.z, 0, fwd.x};
        Vector3 move = {0,0,0};
        bool side = false;

        if (IsKeyDown(KEY_W)) move = Vector3Add(move, fwd);
        if (IsKeyDown(KEY_S)) move = Vector3Subtract(move, fwd);
        if (IsKeyDown(KEY_A)) { move = Vector3Subtract(move, rgt); flipX = true; side = true; }
        if (IsKeyDown(KEY_D)) { move = Vector3Add(move, rgt); flipX = false; side = true; }
        
        // --- ЛОГИКА ТАСКАНИЯ ЯЩИКОВ ---
        if (IsKeyPressed(KEY_E)) {
            if (grabbedProp) {
                // Если уже держим - отпускаем
                grabbedProp->isHeld = false;
                grabbedProp = nullptr;
            } else {
                // Если не держим - ищем ближайший
                for (auto& p : props) {
                    if (Vector3Distance(pos, p.pos) < 2.0f) {
                        grabbedProp = &p;
                        grabbedProp->isHeld = true;
                        break;
                    }
                }
            }
        }

        if (Vector3Length(move) > 0.1f) {
            pos = Vector3Add(pos, Vector3Scale(Vector3Normalize(move), 7.0f * dt));
            state = side ? WALK_SIDE : WALK;
        } else state = IDLE;

        vel.y -= 25.0f * dt;
        pos.y += vel.y * dt;

        bool onGround = false;
        ResolveMapCollision(pos, vel, radius, height, map);
        if (IsKeyPressed(KEY_SPACE) && fabsf(vel.y) < 0.1f) vel.y = 9.0f;
        
        if (pos.y < -10.0f) pos = {0, 5, 0};

        anims[state].Update(dt);
    }

    void Draw(const Camera3D& camera) {
        Texture2D tex = anims[state].Get();
        if (tex.id == 0) return;
        float w = flipX ? -(float)tex.width : (float)tex.width;
        DrawBillboardRec(camera, tex, {0,0,w,(float)tex.height}, {pos.x, pos.y + height/2, pos.z}, {fabsf(w/tex.height)*height, height}, WHITE);
    }
};

// ----------------------------------------------------------------------------------
// МИР ИГРЫ
// ----------------------------------------------------------------------------------
class GameWorld {
private:
    Model level;
    std::unique_ptr<Player> player;
    std::vector<std::unique_ptr<Actor>> actors;
    std::vector<Projectile> projectiles;
    
    // Новые контейнеры (должны быть здесь)
    std::vector<SpawnBase> bases;
    std::vector<Prop> props;

    Camera3D camera = {0};
    
    Shader modelShader;
    int lightPosLoc, viewPosLoc, playerPosLoc;
    RenderTexture2D shadowMap;
    Shader shadowShader;
    int shadowMapLoc;
    int lightSpaceMatrixLoc;
    Matrix lightSpaceMatrix;
    Texture2D fireballTex;
    Sound fireballFly;
    Sound fireballExp;
    Sound radioMusic;
    float globalSpawnTimer = 120.0f;
    const float SPAWN_INTERVAL = 120.0f;

    void LoadBases() {
        FILE* f = fopen("bases.txt", "r");
        if (f) {
            float x, y, z;
            while (fscanf(f, "%f %f %f", &x, &y, &z) == 3) {
                bases.push_back(SpawnBase({x, y, z}));
            }
            fclose(f);
        } else {
            // Если файла нет, создаем дефолтные базы
            bases.push_back(SpawnBase({20, 0, 20}));
            bases.push_back(SpawnBase({-20, 0, -20}));
        }
    }

    void SpawnWave(Vector3 center) {
        for (int i = 0; i < 5; i++) {
            float angle = i * (PI * 2.0f / 5.0f);
            Vector3 spawnPos = { 
                center.x + cosf(angle) * 5.0f, 
                center.y + 1.0f, 
                center.z + sinf(angle) * 5.0f 
            };
            actors.push_back(std::make_unique<Actor>(spawnPos, true, "textures/enemy.png"));
        }
    }

public:
    GameWorld() {
        InitWindow(1280, 720, "SBS2 - Full Version");
        InitAudioDevice(); // Запускает аудио-карту
        SetTargetFPS(60);
        DisableCursor();

        modelShader = LoadShaderFromMemory(vertexShader, fragmentShader);
        lightPosLoc   = GetShaderLocation(modelShader, "lightPos");
        viewPosLoc    = GetShaderLocation(modelShader, "viewPos");
        playerPosLoc  = GetShaderLocation(modelShader, "playerPos");

        // Новые PBR параметры
        int metallicLoc  = GetShaderLocation(modelShader, "metallic");
        int roughnessLoc = GetShaderLocation(modelShader, "roughness");
        int aoLoc        = GetShaderLocation(modelShader, "ao");
        shadowMap = LoadRenderTexture(2048, 2048);
        SetTextureFilter(shadowMap.depth, TEXTURE_FILTER_BILINEAR);
        SetTextureWrap(shadowMap.depth, TEXTURE_WRAP_CLAMP);
shadowShader = LoadShaderFromMemory(shadowVertexShader, shadowFragmentShader);
// Устанавливаем location для MVP матрицы в shadow шейдере
shadowShader.locs[SHADER_LOC_MATRIX_MVP] = GetShaderLocation(shadowShader, "mvp");

// В model shader убедитесь что matModel установлен
modelShader.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocation(modelShader, "matModel");

        shadowMapLoc = GetShaderLocation(modelShader, "shadowMap");
        lightSpaceMatrixLoc = GetShaderLocation(modelShader, "lightSpaceMatrix");
        // Можно задать значения по умолчанию
        float defMetallic = 0.0f, defRoughness = 0.85f, defAO = 1.0f;
        SetShaderValue(modelShader, metallicLoc,  &defMetallic,  SHADER_UNIFORM_FLOAT);
        SetShaderValue(modelShader, roughnessLoc, &defRoughness, SHADER_UNIFORM_FLOAT);
        SetShaderValue(modelShader, aoLoc,        &defAO,        SHADER_UNIFORM_FLOAT);

        level = LoadModel("level.obj");
        if (level.meshCount == 0) level = LoadModelFromMesh(GenMeshPlane(100, 100, 10, 10));
        for (int i = 0; i < level.materialCount; i++) level.materials[i].shader = modelShader;

        if (FileExists("textures/fireball.png")) {
            fireballTex = LoadTexture("textures/fireball.png");
        } else {
            Image img = GenImageColor(32, 32, ORANGE);
            fireballTex = LoadTextureFromImage(img);
            UnloadImage(img);
        }

        player = std::make_unique<Player>((Vector3){0, 2, 0});

        // Инициализация баз и ящиков
        LoadBases();
        fireballFly = LoadSound("sounds/fly.wav");
        fireballExp = LoadSound("sounds/exp.wav");
        radioMusic = LoadSound("sounds/music.mp3");
        props.push_back({{5, 1, 5}, false,false, {0}}); 
        props.push_back({{-5, 1, 8}, false,false, {0}});
        // Сделаем один из пропов радио
        if (props.size() > 0) {
            props[0].isRadio = true;
            props[0].music = radioMusic;
        }


        // Стартовые актеры
        actors.push_back(std::make_unique<Actor>((Vector3){10, 1, 10}, true, "textures/enemy.png"));
        actors.push_back(std::make_unique<Actor>((Vector3){-5, 1, -5}, false, "textures/ally.png"));
        
        camera.up = {0, 1, 0};
        camera.fovy = 70.0f;
    }

    void Run() {
        while (!WindowShouldClose()) {
            float dt = GetFrameTime();
            
            // --- ЛОГИКА ТАЙМЕРА И СПАВНА ---
            globalSpawnTimer -= dt;
            if (globalSpawnTimer <= 0) {
                for (auto& base : bases) {
                    if (!base.isCaptured) SpawnWave(base.position);
                }
                globalSpawnTimer = SPAWN_INTERVAL;
            }

            // --- ОБНОВЛЕНИЕ БАЗ, ЯЩИКОВ И ИГРОКА ---
            for (auto& b : bases) b.UpdateCaptureState(player->pos, actors);
            
            // Передаем props в Player::Update
            player->Update(dt, level, props);

            Vector3 fwd = {-sinf(player->camH), 0, -cosf(player->camH)};
            for (auto& p : props) p.Update(player->pos, fwd, dt, level);

            // --- ОБНОВЛЕНИЕ АКТЕРОВ ---
            for (auto& actor : actors) {
                actor->Update(dt, player->pos, level, actors, projectiles);
                
                // Проверка: Враг коснулся Игрока -> Рестарт
                if (actor->isEnemy && actor->health > 0) {
                    if (Vector3Distance(actor->pos, player->pos) < 1.0f) {
                        player->pos = {0, 5, 0};
                    }
                }
            }

            // --- ОБНОВЛЕНИЕ ПУЛЬ ---
            for (int i = projectiles.size() - 1; i >= 0; i--) {
                Vector3 oldPos = projectiles[i].pos;
                projectiles[i].Update(dt, level,player->pos, fireballFly, fireballExp);
                Vector3 newPos = projectiles[i].pos;

                if (projectiles[i].active) {
                    for (auto& actor : actors) {
                        if (actor->isEnemy && actor->health > 0) {
                            Vector3 hitboxCenter = Vector3Add(actor->pos, { 0, 1.0f, 0 });
                            Vector3 moveVec = Vector3Subtract(newPos, oldPos);
                            float stepDist = Vector3Length(moveVec);
                            Ray trajectory = { oldPos, Vector3Normalize(moveVec) };

                            RayCollision hit = GetRayCollisionSphere(trajectory, hitboxCenter, 1.2f);

                            if (hit.hit && hit.distance <= stepDist + 0.2f) {
                                actor->health -= 100;
                                projectiles[i].active = false;
                                break;
                            }
                        }
                    }
                }
                if (!projectiles[i].active) {
                    projectiles.erase(projectiles.begin() + i);
                }
            }
                        // ==================== SHADOW MAP PASS ====================
// ==================== SHADOW MAP PASS ====================
// ==================== SHADOW MAP PASS ====================
Vector3 sunDir = Vector3Normalize({ -0.4f, -0.9f, -0.3f });
Vector3 lightPos = Vector3Scale(Vector3Negate(sunDir), 50.0f);
Vector3 lightTarget = player->pos;

// Ортографическая проекция для теней
Matrix lightProj = MatrixOrtho(-50.0f, 50.0f, -50.0f, 50.0f, 1.0f, 200.0f);
Matrix lightView = MatrixLookAt(lightPos, lightTarget, {0, 1, 0});
lightSpaceMatrix = MatrixMultiply(lightView, lightProj);

// Рендерим в текстуру теней
BeginTextureMode(shadowMap);
    // В shadow pass нам нужна только глубина, отключаем цвет
    rlSetCullFace(RL_CULL_FACE_FRONT); // Используем фронтальное отсечение для уменьшения acne
    
    BeginMode3D((Camera3D){lightPos, lightTarget, {0, 1, 0}, 90.0f, 0});
        BeginShaderMode(shadowShader);
            // Рендерим уровень (только геометрия)
            for (int i = 0; i < level.meshCount; i++) {
                DrawMesh(level.meshes[i], level.materials[0], level.transform);
            }
            
            // Рендерим ящики
            for (auto& p : props) {
                DrawCube(p.pos, 1.0f, 1.0f, 1.0f, WHITE);
            }
            
            // Рендерим актеров
            for (auto& actor : actors) {
                if (actor->health > 0) {
                    DrawCube(Vector3Add(actor->pos, {0, 1.0f, 0}), 1.0f, 2.0f, 1.0f, WHITE);
                }
            }
            
            // Рендерим игрока
            DrawCube(Vector3Add(player->pos, {0, 1.0f, 0}), 1.0f, 2.0f, 1.0f, WHITE);
            
        EndShaderMode();
    EndMode3D();
    
    rlSetCullFace(RL_CULL_FACE_BACK); // Возвращаем обычное отсечение
EndTextureMode();
// ==================== КОНЕЦ SHADOW PASS ====================
            // ==================== КОНЕЦ SHADOW PASS ====================
            
            // --- КАМЕРА ---
            // ==================== КОНЕЦ SHADOW PASS ====================

// ПЕРЕДАЕМ МАТРИЦУ И ТЕКСТУРУ В ОСНОВНОЙ ШЕЙДЕР
SetShaderValueMatrix(modelShader, lightSpaceMatrixLoc, lightSpaceMatrix);

// Привязываем текстуру теней к слоту 15

// --- КАМЕРА ---
Vector3 head = Vector3Add(player->pos, {0, 1.8f, 0});
// ... остальной код
            Vector3 offset = { cosf(player->camV) * sinf(player->camH), sinf(player->camV), cosf(player->camV) * cosf(player->camH) };
            float dist = 7.0f;
            Ray camRay = { head, offset };
            for(int i=0; i<level.meshCount; i++) {
                RayCollision c = GetRayCollisionMesh(camRay, level.meshes[i], level.transform);
                if (c.hit && c.distance < dist) dist = c.distance - 0.2f;
            }
            camera.position = Vector3Add(head, Vector3Scale(offset, dist));
            camera.target = head;

            // --- ШЕЙДЕР ---
            Vector3 sunLightPos = {20, 40, 10};
            SetShaderValue(modelShader, lightPosLoc, &sunLightPos, SHADER_UNIFORM_VEC3);
            SetShaderValue(modelShader, viewPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
            SetShaderValue(modelShader, playerPosLoc, &player->pos, SHADER_UNIFORM_VEC3);
            SetShaderValueMatrix(modelShader, lightSpaceMatrixLoc, lightSpaceMatrix);

            // Привязываем текстуру теней к текстурному слоту 1
            int shadowTextureUnit = 1;
            rlActiveTextureSlot(shadowTextureUnit);
            rlEnableTexture(shadowMap.depth.id);
            SetShaderValue(modelShader, shadowMapLoc, &shadowTextureUnit, SHADER_UNIFORM_INT);
            rlActiveTextureSlot(0); // Возвращаемся к основному слоту

            // --- ОТРИСОВКА ---
            BeginDrawing();

                ClearBackground({20, 20, 25, 255});
                
                BeginMode3D(camera);
                    DrawModel(level, {0,0,0}, 1.0f, WHITE);
                     // Выключаем шейдер для спрайтов и линий
                    rlEnableColorBlend(); 
                    rlDisableBackfaceCulling(); // Чтобы видеть спрайт с обеих сторон
                    player->Draw(camera);
                    for (auto& actor : actors) actor->Draw(camera);
                    
                    for (auto& p : projectiles) DrawBillboard(camera, fireballTex, p.pos, 0.5f, YELLOW);
                    for (auto& p : props) p.Draw();
                    for (auto& b : bases) DrawCircle3D(b.position, 10.0f, {1,0,0}, 90, b.isCaptured ? GREEN : RED);
                EndShaderMode();
                EndMode3D();
                DrawTexturePro(shadowMap.depth, 
                    (Rectangle){0, 0, 2048, 2048}, 
                    (Rectangle){20, 400, 256, 256}, 
                    (Vector2){0, 0}, 0.0f, WHITE);
                
                DrawText("Shadow Map (depth)", 30, 370, 20, YELLOW);
                // UI (РИСУЕТСЯ ПОВЕРХ 3D)
                DrawText("WASD - Move | E - Grab/Ally | Shift - Run", 10, 10, 20, GREEN);
                DrawText(TextFormat("Projectiles: %i | Actors: %i", (int)projectiles.size(), (int)actors.size()), 10, 40, 20, RAYWHITE);
                DrawText(TextFormat("Wave Timer: %.1f", globalSpawnTimer), 10, 70, 20, YELLOW);
                if (player->grabbedProp) DrawText("CARRYING OBJECT", 10, 100, 20, ORANGE);
                // --- DEBUG OVERLAY ---
                DrawRectangle(10, 150, 300, 150, Fade(BLACK, 0.7f)); // Фон для читаемости
                DrawText("DEBUG ACTORS:", 20, 160, 20, YELLOW);

                for (int i = 0; i < actors.size() && i < 5; i++) { // Выведем инфо по первым 5 актерам
                    Color debugColor = actors[i]->isEnemy ? RED : GREEN;
                    const char* type = actors[i]->isEnemy ? "Enemy" : "Ally";
                    
                    DrawText(TextFormat("#%i %s | HP: %.0f", i, type, actors[i]->health), 
                            20, 185 + (i * 25), 18, debugColor);
                            
                    DrawText(TextFormat("Pos: %.1f, %.1f, %.1f", 
                            actors[i]->pos.x, actors[i]->pos.y, actors[i]->pos.z), 
                            140, 185 + (i * 25), 18, RAYWHITE);
                }
                DrawFPS(1200, 10);
            EndDrawing();
        }
    }

    ~GameWorld() { 
        CloseAudioDevice(); // Корректно закрывает звук при выходе
        UnloadSound(fireballFly);
        UnloadSound(fireballExp);
        UnloadSound(radioMusic);
        UnloadShader(modelShader); 
        UnloadModel(level); 
        UnloadTexture(fireballTex);
        CloseWindow(); 
    }
};

int main() { GameWorld w; w.Run(); return 0; }