import {defs, tiny} from './examples/common.js'; 

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene, Texture
} = tiny;



//from https://github.com/Robert-Lu/tiny-graphics-shadow_demo/blob/master/examples/obj-file-demo.js?fbclid=IwAR2t3SJ42QS89LvHg1X9yfPLjp80O1ThklxkqSnwqV6dS7D0jK27s0z6pBA
export class Shape_From_File extends Shape {                                   // **Shape_From_File** is a versatile standalone Shape that imports
                                                                               // all its arrays' data from an .obj 3D model file.
    constructor(filename) {
        super("position", "normal", "texture_coord");
        // Begin downloading the mesh. Once that completes, return
        // control to our parse_into_mesh function.
        this.load_file(filename);
    }

    load_file(filename) {                             // Request the external file and wait for it to load.
        // Failure mode:  Loads an empty shape.
        return fetch(filename)
            .then(response => {
                if (response.ok) return Promise.resolve(response.text())
                else return Promise.reject(response.status)
            })
            .then(obj_file_contents => this.parse_into_mesh(obj_file_contents))
            .catch(error => {
                this.copy_onto_graphics_card(this.gl);
            })
    }

    parse_into_mesh(data) {                           // Adapted from the "webgl-obj-loader.js" library found online:
        var verts = [], vertNormals = [], textures = [], unpacked = {};

        unpacked.verts = [];
        unpacked.norms = [];
        unpacked.textures = [];
        unpacked.hashindices = {};
        unpacked.indices = [];
        unpacked.index = 0;

        var lines = data.split('\n');

        var VERTEX_RE = /^v\s/;
        var NORMAL_RE = /^vn\s/;
        var TEXTURE_RE = /^vt\s/;
        var FACE_RE = /^f\s/;
        var WHITESPACE_RE = /\s+/;

        for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            var elements = line.split(WHITESPACE_RE);
            elements.shift();

            if (VERTEX_RE.test(line)) verts.push.apply(verts, elements);
            else if (NORMAL_RE.test(line)) vertNormals.push.apply(vertNormals, elements);
            else if (TEXTURE_RE.test(line)) textures.push.apply(textures, elements);
            else if (FACE_RE.test(line)) {
                var quad = false;
                for (var j = 0, eleLen = elements.length; j < eleLen; j++) {
                    if (j === 3 && !quad) {
                        j = 2;
                        quad = true;
                    }
                    if (elements[j] in unpacked.hashindices)
                        unpacked.indices.push(unpacked.hashindices[elements[j]]);
                    else {
                        var vertex = elements[j].split('/');

                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 0]);
                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 1]);
                        unpacked.verts.push(+verts[(vertex[0] - 1) * 3 + 2]);

                        if (textures.length) {
                            unpacked.textures.push(+textures[((vertex[1] - 1) || vertex[0]) * 2 + 0]);
                            unpacked.textures.push(+textures[((vertex[1] - 1) || vertex[0]) * 2 + 1]);
                        }

                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 0]);
                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 1]);
                        unpacked.norms.push(+vertNormals[((vertex[2] - 1) || vertex[0]) * 3 + 2]);

                        unpacked.hashindices[elements[j]] = unpacked.index;
                        unpacked.indices.push(unpacked.index);
                        unpacked.index += 1;
                    }
                    if (j === 3 && quad) unpacked.indices.push(unpacked.hashindices[elements[0]]);
                }
            }
        }
        {
            const {verts, norms, textures} = unpacked;
            for (var j = 0; j < verts.length / 3; j++) {
                this.arrays.position.push(vec3(verts[3 * j], verts[3 * j + 1], verts[3 * j + 2]));
                this.arrays.normal.push(vec3(norms[3 * j], norms[3 * j + 1], norms[3 * j + 2]));
                this.arrays.texture_coord.push(vec(textures[2 * j], textures[2 * j + 1]));
            }
            this.indices = unpacked.indices;
        }
        this.normalize_positions(false);
        this.ready = true;
    }

    draw(context, program_state, model_transform, material) {               // draw(): Same as always for shapes, but cancel all
        // attempts to draw the shape before it loads:
        if (this.ready)
            super.draw(context, program_state, model_transform, material);
    }
}

const origin = vec3(0, 0, 0);

let origin_relative = vec3(0, 0, 0);

let ihat = vec3(1, 0, 0);
let khat = vec3(0, 0, 1);
let z_rot = 0;

function distance_between(a,b){
    return (Math.sqrt( ((a[0]-b[0])*(a[0]-b[0])) + ((a[2]-b[2])*(a[2]-b[2])) ))

}

const wm = class worldmovement extends defs.Movement_Controls
{
  constructor()
  {
    super();
  }

  fpv(radians_per_frame, meters_per_frame, leeway=100)
  {
    

    let foo = vec3(NaN, NaN, NaN);
    Object.assign(foo, origin_relative);

    if (this.thrust[0] !== 0) {
      foo[0] += 1 * this.thrust[0] * ihat[0] * .1;
      foo[2] += 1 * this.thrust[2] * ihat[2] * .1;
    }
    if (this.thrust[2] !== 0) {
      foo[0] += 1 * this.thrust[2] * khat[0] * .1;
      foo[2] += 1 * this.thrust[2] * khat[2] * .1;
    }

    
    const ppdist = (p, q) => {return Math.sqrt(Math.pow(p[0] - q[0], 2) + Math.pow(p[2] - q[2], 2));};
    let cur_dist = ppdist(origin_relative, origin);
    let future_dist = ppdist(foo, origin);

    Object.assign(origin_relative, foo);
  }

 

  display(context, graphics_state)
  {

    if (this.will_take_over_graphics_state)
    {
      this.reset(graphics_state);
      this.will_take_over_graphics_state = false;
    }

    this.fpv((graphics_state.animation_delta_time / 1000) * this.speed_multiplier * this.radians_per_frame, (graphics_state.animation_delta_time / 1000) * this.speed_multiplier * this.meters_per_frame);


  }
};

export class Assignment extends Scene {
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        this.contact_time = 0;
        this.contact_complete = 0;
        this.flag = 0;
        this.r_flag = 0;
        this.delay = 4;
        this.u = 12;
    

        

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            cube: new defs.Cube(),
            torus: new defs.Torus(70, 20),
            torus2: new defs.Torus(3, 150),
            sphere: new defs.Subdivision_Sphere(4),
            circle: new defs.Regular_2D_Polygon(1, 15),
            planet_1: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version()) (2),
            planet_2: new defs.Subdivision_Sphere(3),
            moon: new (defs.Subdivision_Sphere.prototype.make_flat_shaded_version()) (1),
            ground: new defs.Capped_Cylinder(80,80, [[0, 2], [0, 1]]),
            skybox_night: new defs.Subdivision_Sphere(4),
            rocket: new Shape_From_File( "/assets/rocket.obj" ),
                         
        };
        this.shapes.ground.arrays.texture_coord.forEach(f => f.scale_by(60));

        // *** Materials
        this.materials = {
            test: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            test2: new Material(new Gouraud_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#992828")}),
            ring: new Material(new Ring_Shader(), {ambient:1, diffusivity: 1, color: hex_color("#f000ff"), specularity:1}),
            ring2: new Material(new Ring_Shader(), {ambient:1, diffusivity: 1, color: hex_color("#f000ff"), specularity:1}),

            sphere: new Material(new defs.Phong_Shader(),
                {ambient: 1, diffusivity: 1, color: hex_color("#ffa500")}),

            sphere2: new Material(new defs.Phong_Shader(),
                {ambient: 1, diffusivity: 1, color: hex_color("#ADD8E6")}),
            
            matp1: new Material(new defs.Phong_Shader(),
                {ambient:1, diffusivity: 1, color: hex_color("#808080")}), //from https://www.canva.com/colors/color-meanings/gray/

             
            matp21: new Material(new Gouraud_Shader(),
                {diffusivity: .1, color: hex_color("#80ffff"), specularity:1}),
            
            matp22: new Material(new defs.Phong_Shader(),
                {ambient:1, diffusivity: .1, color: hex_color("#80ffff"), specularity:1}),

            matp3: new Material(new defs.Phong_Shader(),
                {ambient:1, diffusivity: 1, color: hex_color("#b08040"), specularity:1}),

            matp4: new Material(new defs.Phong_Shader(),
                {ambient:1,color: hex_color("#c7e4ee"), specularity:1}), //from https://www.color-name.com/soft-light-blue.color
            
            ground: new Material(new defs.Textured_Phong(1), 
                {ambient: 0.5, specularity: 0.1, texture: new Texture("assets/ground2.jpeg")}),

            skybox_night: new Material(new defs.Textured_Phong(1),
                {ambient: 1, specularity: 0.1, color: color(0,0,0,1), texture: new Texture("assets/skyscape.png")}),

            skybox_day: new Material(new defs.Textured_Phong(1),
                {ambient: 1, specularity: 0.1, color: color(0,0,0,1), texture: new Texture("assets/clouds.png")}),
            
        } 

        this.initial_camera_location = Mat4.look_at(vec3(0,-25,10), vec3(0, 10, 7), vec3(0, 1, 1));

        this.night = true;

        this.sparks = [];
        for(var i = 0; i < 300; i++){
            this.sparks.push(new particle());
        }

        this.sparks2 = [];
        for(var i = 0; i < 300; i++){
            this.sparks2.push(new particle());
        }

        this.rocket_contact = 0;
        this.r_flag=0;
        this.firework_time=0;
        this.explosion_flag=0;
        this.spinner_1_flag=0;
        this.spinner_2_flag=0;
        this.spinner_3_flag=0;

        this.fire = new Audio ('Fire.mp3');
        this.explosion = new Audio ('Explosion.mp3');
    }

    make_control_panel() {
        // Draw the scene's buttons, setup their actions and keyboard shortcuts, and monitor live measurements.
        this.key_triggered_button("Toggle between Night and Day", ["n"], () => this.night = !this.night);
        this.new_line();
    }

    sproj(u,t)
    {
        let s = (u*t)+0.5*(-9.81)*Math.pow(t,2);

        return Math.max(s,0);
    }

    vproj(u,t)
    {


        let v = (this.sproj(u,t) +0.5*(-9.81)*Math.pow(t,2))/t

        return v
    }

    display(context, program_state) {
        // display():  Called once per frame of animation.
        // Setup -- This part sets up the scene's overall camera matrix, projection matrix, and lights:
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new wm());
            // Define the global camera and projection matrices, which are stored in program_state.
            //program_state.set_camera(this.initial_camera_location);
            program_state.set_camera(Mat4.look_at(vec3(0, 0, 25), vec3(0, 0, -1), vec3(0, 1, 0)));
        }

        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);

        let cubetransform = Mat4.identity();
        let p2matrix = Mat4.identity();
        let rocket_matrix = Mat4.identity();
        let model_transform = Mat4.identity();
        let ringsmatrix0 = Mat4.identity();
        let ringsmatrix1 = Mat4.identity();
        let ringsmatrix2 = Mat4.identity();
        let ringsmatrix3 = Mat4.identity();

        let p4matrix = Mat4.identity();
        let moonmatrix = Mat4.identity();

        cubetransform = cubetransform.times(Mat4.rotation(Math.PI/23,1,0,0)).times(Mat4.translation(0.5, 2, 21)).times(Mat4.scale(.1,.1,3))
        let cube_pos = vec3(cubetransform[0][3],cubetransform[1][3],cubetransform[2][3]);




        const t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;

        var mod = 0
        if (Math.round(t)%2 == 0)
        {
            mod = 0;
        }
        else if (Math.round(t)%2 == 1)
        {
            mod = 1;
        }
        else
        {
            console.log("Issue in mod")
        }


        let sunmatrix = Mat4.identity();
        var sunscaler = 2 + Math.sin((1/5 * Math.PI * t));

        sunmatrix = sunmatrix.times(Mat4.scale(sunscaler, sunscaler, sunscaler));
         


       


        if(this.rocket_contact==1)
        {
        rocket_matrix = Mat4.identity();
        rocket_matrix = rocket_matrix.times(Mat4.translation(...origin_relative)).times(Mat4.translation(0,this.sproj(this.u,t-this.contact_time-this.delay),0));
        rocket_matrix = rocket_matrix.times(Mat4.scale(.3,.3,.3));
        this.contact_complete = 1;

        }
        else if(this.contact_complete==0)
        {
            rocket_matrix = rocket_matrix.times(Mat4.translation(...origin_relative));
            rocket_matrix = rocket_matrix.times(Mat4.scale(.3,.3,.3));
        }

                 


        
         if(this.spinner_1_flag==1)
         {
            ringsmatrix0 = ringsmatrix0.times(Mat4.translation(...origin_relative)).times(Mat4.translation(6, -1, 0)).times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.scale(0.3,0.3,0.3));
            ringsmatrix3 = ringsmatrix3.times(Mat4.translation(...origin_relative)).times(Mat4.translation(6, 2, 0)).times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.rotation(2*t, 0, 0, 1)).times(Mat4.translation(0, 0 ,3)).times(Mat4.rotation(0.2*Math.sin(10 * Math.PI * t/2), 1, 1, 1)).times(Mat4.scale(1,1,0.0001));
         }
         else
         {
            ringsmatrix0 = ringsmatrix0.times(Mat4.translation(...origin_relative)).times(Mat4.translation(6, -1, 0)).times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.scale(0.3,0.3,0.3));
            ringsmatrix3 = ringsmatrix3.times(Mat4.translation(...origin_relative)).times(Mat4.translation(6, 2, 0)).times(Mat4.rotation(Math.PI/2, 1, 0, 0)).times(Mat4.translation(0, 0 ,3)).times(Mat4.scale(1,1,0.0001));
         }

        let spinner_1_pos = vec3(ringsmatrix0[0][3],ringsmatrix0[1][3],ringsmatrix0[2][3]);
        if(distance_between(cube_pos,spinner_1_pos)<0.9)
        {
            this.spinner_1_flag=1;
            this.fire.play();
        }

        let ringsmatrix4 = Mat4.identity();
        let ringsmatrix5 = Mat4.identity();
        if(this.spinner_2_flag==1)
        {        
            ringsmatrix4 = ringsmatrix0.times(Mat4.translation(13, 0, 0)).times(Mat4.rotation(6*t, 0, 0, 1)).times(Mat4.translation(2, 0, 0)).times(Mat4.scale(1.2,1.2,1.2));
            ringsmatrix5 = ringsmatrix0.times(Mat4.translation(13, 0, 0)).times(Mat4.rotation(6*t, 0, 0, 1)).times(Mat4.translation(2, 0 , 0)).times(Mat4.rotation(0.1*Math.sin(7 * Math.PI * t/2), 1, 1, 1)).times(Mat4.scale(2,2,0.0001));
        }
        else
        {         
            ringsmatrix4 = ringsmatrix0.times(Mat4.translation(13, 0, 0)).times(Mat4.translation(2, 0, 0)).times(Mat4.scale(1.2,1.2,1.2));
            ringsmatrix5 = ringsmatrix0.times(Mat4.translation(13, 0, 0)).times(Mat4.translation(2, 0 , 0)).times(Mat4.scale(2,2,0.0001));
        }

        let spinner_2_pos = vec3(ringsmatrix4[0][3],ringsmatrix4[1][3],ringsmatrix4[2][3]);
        if(distance_between(cube_pos,spinner_2_pos)<0.9)
        {
            this.spinner_2_flag=1;
            this.fire.play();
        }
          
        this.shapes.torus.draw(context, program_state, ringsmatrix0, this.materials.ring2);
        this.shapes.torus.draw(context, program_state, ringsmatrix3, this.materials.ring);

        this.shapes.torus.draw(context, program_state, ringsmatrix4, this.materials.ring2);
        this.shapes.torus.draw(context, program_state, ringsmatrix5, this.materials.ring);

        const light_position = vec4(0,0, 0 , 1);
        let light = [new Light(light_position, color(1,  0.5 + 0.5*Math.sin(1/5 * Math.PI * t),  0.5 + 0.5*Math.sin(1/5 * Math.PI * t),1), 10 **sunscaler)]
        program_state.lights = light;


        const red = hex_color("#d40402");
        const white = hex_color("#ffffff");
        const blue = hex_color("#1738B7");
        

         if(this.rocket_contact==1){
         if (this.vproj(this.u, t-this.contact_time-this.delay)>0)
         {
             this.shapes.rocket.draw(context, program_state, rocket_matrix, this.materials.matp3);
         }
         else if(this.explosion_flag==0)
         {
             this.firework_time=t;
             this.flag=1;
             this.explosion_flag=1;
             this.explosion.play();
             this.fire.play();
         }
         }
         else{
             this.shapes.rocket.draw(context, program_state, rocket_matrix, this.materials.matp3);
         }


        this.shapes.cube.draw(context, program_state, cubetransform, this.materials.matp1);


        let rocket_pos = vec3(rocket_matrix[0][3],rocket_matrix[1][3],rocket_matrix[2][3]);

        if(distance_between(cube_pos,rocket_pos)<0.7 && this.r_flag==0)
        {
            this.rocket_contact=1;
            this.contact_time = t;
            this.r_flag = 1;
            this.fire.play();

        }

        let ground_t = Mat4.identity().times(Mat4.translation(...origin_relative))
                                      .times(Mat4.rotation(z_rot, 0, 1, 0))
                                      .times(Mat4.translation(...origin))
                                      .times(Mat4.rotation(Math.PI/2, 1, 0, 0))
                                      .times(Mat4.translation(0, 0, 2))
                                      .times(Mat4.scale(60, 60, 0.5));
        
        

        let sky_t = Mat4.identity().times(Mat4.rotation(z_rot, 0, 1, 0))
                                      .times(Mat4.translation(...origin))
                                      .times(Mat4.rotation(Math.PI/2, 1, 0, 0))
                                      .times(Mat4.scale(70, 70, 70))
                                      .times(Mat4.rotation(t/50, 0, 1, 0));
        
        if (this.night){
            this.shapes.skybox_night.draw(context, program_state, sky_t, this.materials.skybox_night);
            this.shapes.ground.draw(context, program_state, ground_t, this.materials.ground);
        }
        else{
            this.shapes.skybox_night.draw(context, program_state, sky_t, this.materials.skybox_day);
            this.shapes.ground.draw(context, program_state, ground_t, this.materials.ground.override({ambient: 1}));
        }

        
        
        if(this.attached != undefined)
        {

             program_state.camera_inverse = this.attached().map((x,i) => Vector.from(program_state.camera_inverse[i]).mix(x, 0.1));
        }

        if(this.flag==1){
        for(var i = 0; i < 300; i++){
            model_transform = Mat4.identity();
            model_transform = model_transform.times(Mat4.translation(...origin_relative)).times(Mat4.translation(0,7.35,0)).times(Mat4.scale(1/35,1/35,1/35));;       
            this.sparks[i].position = (this.sparks[i].velocity.times(1)).plus(this.sparks[i].position);
            this.sparks[i].velocity = (this.sparks[i].acceleration.times(0.4*Math.random())).plus(this.sparks[i].velocity);
            this.sparks[i].velocity = (this.sparks[i].velocity).times(0.7)
            model_transform = model_transform.times(Mat4.translation(this.sparks[i].position[0],this.sparks[i].position[1],this.sparks[i].position[2]));
            this.sparks[i].trans = model_transform;
            if(t-this.firework_time+(7.5*Math.random())>=10){
                this.sparks[i].life=0;
            }
        }
        


        for(var i = 0; i < 300; i++){
        if(this.sparks[i].life){
        this.shapes.sphere.draw(context,program_state,this.sparks[i].trans,this.materials.matp3.override({color:blue}));
        }
        }
        }

       for(var i = 0; i < 300; i++){
            model_transform = Mat4.identity();
            model_transform = model_transform.times(Mat4.translation(...origin_relative)).times(Mat4.translation(-6,6,7)).times(Mat4.scale(1/70,1/70,1/70));       
            this.sparks2[i].position = (this.sparks2[i].velocity.times(dt)).plus(this.sparks2[i].position);
            this.sparks2[i].velocity = (this.sparks2[i].acceleration.times(dt)).plus(this.sparks2[i].velocity);
            //model_transform = model_transform.times(Mat4.translation())
            model_transform = model_transform.times(Mat4.translation(this.sparks2[i].position[0],this.sparks2[i].position[1],this.sparks2[i].position[2]));
            this.sparks2[i].trans = model_transform;
           if(t+(7*Math.random())>=10){
                this.sparks2[i].life=0;
            }

        }

        for(var i = 0; i < 300; i++){
        if(this.sparks2[i].life)
        this.shapes.sphere.draw(context,program_state,this.sparks2[i].trans,this.materials.matp3.override({color:white}));
        }



    }
}

class Gouraud_Shader extends Shader {
    // This is a Shader using Phong_Shader as template
    // TODO: Modify the glsl coder here to create a Gouraud Shader (Planet 2)

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;

        // Specifier "varying" means a variable's final value will be passed from the vertex shader
        // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
        // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        varying vec4 vc;
        //varying vec3vc;
        varying vec3 N, vertex_worldspace;

        // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
            // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            for(int i = 0; i < N_LIGHTS; i++){
                // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                // light will appear directional (uniform direction from all points), and we 
                // simply obtain a vector towards the light by directly using the stored value.
                // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                // the point light's location from the current surface point.  In either case, 
                // fade (attenuate) the light as the vector needed to reach it gets longer.  
                vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                               light_positions_or_vectors[i].w * vertex_worldspace;                                             
                float distance_to_light = length( surface_to_light_vector );

                vec3 L = normalize( surface_to_light_vector );
                vec3 H = normalize( L + E );
                // Compute the diffuse and specular components from the Phong
                // Reflection Model, using Blinn's "halfway vector" method:
                float diffuse  =      max( dot( N, L ), 0.0 );
                float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                
                vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                          + light_colors[i].xyz * specularity * specular;
                result += attenuation * light_contribution;
            }
            return result;
        } `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;                            
            // Position is expressed in object coordinates.
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                // The final normal vector in screen space.
                N = normalize( mat3( model_transform ) * normal / squared_scale);
                vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;


                vc = vec4(ambient*shape_color.xyz, shape_color.w);

                vc.xyz = vc.xyz+ phong_model_lights(normalize(N), vertex_worldspace);

            } `;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // A fragment is a pixel that's overlapped by the current triangle.
        // Fragments affect the final image or get discarded due to depth.
        return this.shared_glsl_code() + `
            void main(){                                                           
                // Compute an initial (ambient) color:
                
                gl_FragColor = vc;
                //gl_FragColor = hex_color("#b08040");
                // Compute the final color with contributions from lights:
                gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
            } `;
    }

    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
    }

    send_gpu_state(gl, gpu, gpu_state, model_transform) {
        // send_gpu_state():  Send the state of our whole drawing context to the GPU.
        const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
        gl.uniform3fv(gpu.camera_center, camera_center);
        // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
        const squared_scale = model_transform.reduce(
            (acc, r) => {
                return acc.plus(vec4(...r).times_pairwise(r))
            }, vec4(0, 0, 0, 0)).to3();
        gl.uniform3fv(gpu.squared_scale, squared_scale);
        // Send the current matrices to the shader.  Go ahead and pre-compute
        // the products we'll need of the of the three special matrices and just
        // cache and send those.  They will be the same throughout this draw
        // call, and thus across each instance of the vertex shader.
        // Transpose them since the GPU expects matrices as column-major arrays.
        const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
        gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Omitting lights will show only the material color, scaled by the ambient term:
        if (!gpu_state.lights.length)
            return;

        const light_positions_flattened = [], light_colors_flattened = [];
        for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
            light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
            light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
        }
        gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
        gl.uniform4fv(gpu.light_colors, light_colors_flattened);
        gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
    }

    update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
        // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
        // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
        // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
        // program (which we call the "Program_State").  Send both a material and a program state to the shaders
        // within this function, one data field at a time, to fully initialize the shader for a draw.

        // Fill in any missing fields in the Material object with custom defaults for this shader:
        const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
        material = Object.assign({}, defaults, material);

        this.send_material(context, gpu_addresses, material);
        this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
    }
}

class Ring_Shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return `
        precision mediump float;
        varying vec4 point_position;
        varying vec4 center;
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        // TODO:  Complete the main function of the vertex shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        attribute vec3 position;
        uniform mat4 model_transform;
        uniform mat4 projection_camera_model_transform;
        
        void main(){
            


            point_position = vec4(position,1);


            gl_Position = projection_camera_model_transform*point_position;
            
            center = vec4(0,0,position.z,1);
              
        }`;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // TODO:  Complete the main function of the fragment shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        void main(){

          //gl_FragColor= vec4(0.75,0.5,0.25,sin(distance(center,point_position)*55.0) + 0.5); OLD FRAG COLOR

          gl_FragColor = vec4(sin(distance(center,point_position)*48.0) + 0.5,sin(distance(center,point_position)*35.0) + 0.5,0.7,sin(distance(center,point_position)*20.0) + 0.5);
          
        }`;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

class particle
{
    constructor(){
    this.color=vec4(1,0,0,1);
    this.position=vec3(0,0,0);
    this.velocity=vec3(7*Math.random()+1,7*Math.random()+1,7*Math.random()+1);
    this.acceleration=vec3(-1+2*Math.random(),-1.5,-1+2*Math.random());
    this.mass=1;
    this.trans=Mat4.identity();
    this.life=1;
    }
}