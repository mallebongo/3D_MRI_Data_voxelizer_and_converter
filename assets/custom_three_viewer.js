// assets/custom_three_viewer.js

// Ensure THREE is available (it should be if Dash loads assets correctly after CDN)
if (typeof THREE === 'undefined') {
    console.error("THREE.js not loaded. Make sure it's included before this script.");
}

// Global Three.js variables for the MRI viewer
let mriScene, mriCamera, mriRenderer, mriControls;
let mriPointCloud;
let mriIsRotating = false;
const mriCanvasContainerId = 'threejs-canvas-container';
const mriCanvasId = 'threejs-canvas';

// Namespace for our functions to avoid polluting global scope
window.MRIViewer = {
    initScene: function() {
        console.log("MRIViewer.initScene called");
        const container = document.getElementById(mriCanvasContainerId);
        const canvas = document.getElementById(mriCanvasId);

        if (!container || !canvas) {
            console.error("Canvas container or canvas element not found for Three.js.");
            return;
        }

        mriScene = new THREE.Scene();
        mriScene.background = new THREE.Color(0x111111); // Dark background

        const aspect = container.clientWidth / container.clientHeight;
        mriCamera = new THREE.PerspectiveCamera(75, aspect, 0.1, 2000); // Increased far plane
        mriCamera.position.set(0, 0, 150); // Default camera position

        mriRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        mriRenderer.setSize(container.clientWidth, container.clientHeight);
        mriRenderer.setPixelRatio(window.devicePixelRatio);

        mriControls = new THREE.OrbitControls(mriCamera, mriRenderer.domElement);
        mriControls.enableDamping = true;
        mriControls.dampingFactor = 0.05;
        mriControls.minDistance = 10;
        mriControls.maxDistance = 1000;


        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        mriScene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        mriScene.add(directionalLight);
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-50, -50, -50);
        mriScene.add(directionalLight2);


        // Axes Helper - useful for orientation
        // const axesHelper = new THREE.AxesHelper(100);
        // mriScene.add(axesHelper);


        window.addEventListener('resize', this.onWindowResize, false);
        this.animate();
        console.log("Three.js scene initialized.");
    },

    onWindowResize: function() {
        const container = document.getElementById(mriCanvasContainerId);
        if (mriCamera && mriRenderer && container) {
            mriCamera.aspect = container.clientWidth / container.clientHeight;
            mriCamera.updateProjectionMatrix();
            mriRenderer.setSize(container.clientWidth, container.clientHeight);
        }
    },

    animate: function() {
        requestAnimationFrame(window.MRIViewer.animate); // Ensure correct 'this' or use arrow function
        if (mriControls) mriControls.update();
        if (mriIsRotating && mriPointCloud) {
            mriPointCloud.rotation.y += 0.005;
            mriPointCloud.rotation.x += 0.001; // Gentle x rotation
        }
        if (mriRenderer && mriScene && mriCamera) {
            mriRenderer.render(mriScene, mriCamera);
        }
    },

    toggleRotation: function() {
        mriIsRotating = !mriIsRotating;
        // Optionally, update a button text if you add a button for this in Dash
        console.log("Rotation toggled:", mriIsRotating);
        return mriIsRotating; // Return status for potential Dash callback
    },

    resetView: function() {
        if (mriCamera && mriControls) {
            // Find the center of the point cloud if it exists
            let target = new THREE.Vector3(0, 0, 0);
            if (mriPointCloud && mriPointCloud.geometry.boundingSphere) {
                target = mriPointCloud.geometry.boundingSphere.center.clone();
            }
            
            // Estimate a good distance based on bounding box/sphere if available
            // This is a heuristic, adjust as needed
            let distance = 150; 
            if (mriPointCloud && mriPointCloud.geometry.boundingBox) {
                const size = new THREE.Vector3();
                mriPointCloud.geometry.boundingBox.getSize(size);
                distance = Math.max(size.x, size.y, size.z) * 1.5;
                if (distance < 50) distance = 150; // Min distance
            }

            mriCamera.position.set(target.x, target.y, target.z + distance);
            mriCamera.lookAt(target);
            mriControls.target.copy(target);
            mriControls.update();
            console.log("View reset.");
        }
    },

    // Color mapping functions (from standalone.html, adapted)
    getColorRainbow: function(value) {
        const hue = (1 - value) * 0.7;
        return new THREE.Color().setHSL(hue, 1, 0.5);
    },
    getColorGrayscale: function(value) {
        return new THREE.Color(value, value, value);
    },
    getColorHeatmap: function(value) {
        let r, g, b;
        if (value < 0.33) { r = value * 3; g = 0; b = 0; }
        else if (value < 0.66) { r = 1; g = (value - 0.33) * 3; b = 0; }
        else { r = 1; g = 1; b = (value - 0.66) * 3; }
        return new THREE.Color(r, g, b);
    },

    getColor: function(value, scheme) {
        switch (scheme) {
            case 'grayscale': return this.getColorGrayscale(value);
            case 'heatmap': return this.getColorHeatmap(value);
            case 'rainbow':
            default: return this.getColorRainbow(value);
        }
    },

    updatePointCloud: function(voxelData, vizOptions, dataMetadata) {
        // voxelData: array of [x, y, z, value]
        // vizOptions: { colorScheme: 'rainbow', pointSize: 1.0, threshold: 0.1, invertCutoff: false,
        //                spacingX: 1.0, spacingY: 1.0, spacingZ: 1.0 }
        // dataMetadata: { dimensions: [w,h,d], filename: "...", ... }

        console.log("MRIViewer.updatePointCloud called with " + (voxelData ? voxelData.length : 0) + " potential points.");
        console.log("Viz Options:", vizOptions);
        console.log("Data Metadata:", dataMetadata);


        if (mriPointCloud) {
            mriScene.remove(mriPointCloud);
            if (mriPointCloud.geometry) mriPointCloud.geometry.dispose();
            if (mriPointCloud.material) mriPointCloud.material.dispose();
            mriPointCloud = null;
        }

        if (!voxelData || voxelData.length === 0) {
            console.log("No voxel data to display.");
            if (mriRenderer && mriScene && mriCamera) mriRenderer.render(mriScene, mriCamera); // Render empty scene
            return;
        }

        const positions = [];
        const colors = [];

        let actualWidth = 1, actualHeight = 1, actualDepth = 1;
        if (dataMetadata && dataMetadata.dimensions) {
            [actualWidth, actualHeight, actualDepth] = dataMetadata.dimensions;
        }
        
        // Calculate offsets for centering based on original dimensions scaled by spacing
        // These are the dimensions of the *bounding box* of the voxel indices, not the physical size
        const offsetX = -(actualWidth / 2.0) * vizOptions.spacingX;
        const offsetY = -(actualHeight / 2.0) * vizOptions.spacingY;
        const offsetZ = -(actualDepth / 2.0) * vizOptions.spacingZ;

        for (let i = 0; i < voxelData.length; i++) {
            const point = voxelData[i]; // point is [original_i, original_j, original_k, normalized_value]
            const val = point[3];

            // Apply thresholding (Python side should already do this, but double check or make consistent)
            // For now, assume Python sends pre-filtered data based on threshold and inversion
            // If not, add filtering logic here:
            // if ((vizOptions.invertCutoff && val > vizOptions.threshold) || (!vizOptions.invertCutoff && val < vizOptions.threshold)) {
            //     continue;
            // }

            // Apply spacing and offset
            // Original voxel indices are point[0], point[1], point[2]
            positions.push(
                point[0] * vizOptions.spacingX + offsetX,
                point[1] * vizOptions.spacingY + offsetY,
                point[2] * vizOptions.spacingZ + offsetZ
            );

            const color = this.getColor(val, vizOptions.colorScheme);
            colors.push(color.r, color.g, color.b);
        }
        
        console.log("Processed " + (positions.length / 3) + " points for Three.js rendering.");

        if (positions.length === 0) {
            console.log("No points after processing for Three.js.");
            if (mriRenderer && mriScene && mriCamera) mriRenderer.render(mriScene, mriCamera);
            return;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeBoundingSphere(); // Important for controls and camera
        geometry.computeBoundingBox();


        const material = new THREE.PointsMaterial({
            size: parseFloat(vizOptions.pointSize) || 1.0,
            vertexColors: true,
            sizeAttenuation: true, // Points get smaller further away
            // transparent: true, // Enable if opacity < 1
            // opacity: 0.8      // Example opacity
        });

        mriPointCloud = new THREE.Points(geometry, material);
        mriScene.add(mriPointCloud);
        
        // Adjust camera to fit the new point cloud
        // This is a simple fit, might need refinement
        if (mriPointCloud.geometry.boundingSphere) {
             const sphere = mriPointCloud.geometry.boundingSphere;
             const distance = sphere.radius / Math.sin(Math.PI / 180.0 * mriCamera.fov / 2.0);
             
             mriControls.target.copy(sphere.center);
             mriCamera.position.copy(sphere.center).add(new THREE.Vector3(0,0, Math.max(distance, 50) * 1.2 )); // Ensure min distance
             mriCamera.lookAt(sphere.center);
             mriControls.update();
        } else {
            this.resetView(); // Fallback
        }
         console.log("Point cloud updated in Three.js.");
    }
};

// Small utility to ensure the init function is called only once.
(function() {
    let initialized = false;
    const ensureInitialized = () => {
        if (!initialized && document.getElementById(mriCanvasContainerId) && document.getElementById(mriCanvasId)) {
            MRIViewer.initScene();
            initialized = true;
        } else if (!initialized) {
            // If elements not ready, try again shortly
            // console.log("Canvas elements not ready, retrying init...");
            setTimeout(ensureInitialized, 100);
        }
    };
    // Wait for DOM content to be loaded, or at least for Dash to render the layout.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", ensureInitialized);
    } else {
        ensureInitialized(); // DOM is already loaded
    }
})();