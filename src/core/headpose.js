/**
 * Estimates head pose (yaw, pitch, roll) from BlazeFace 6 keypoints.
 * All angles in degrees.
 *
 * BlazeFace keypoint order:
 *   0 = right_eye, 1 = left_eye, 2 = nose_tip,
 *   3 = mouth, 4 = right_ear, 5 = left_ear
 *
 * @param {Array<{x: number, y: number}>} kps - 6 BlazeFace keypoints
 * @returns {{ yaw: number, pitch: number, roll: number, normal: {x: number, y: number, z: number} }}
 */
export function estimateHeadPose(kps) {
    const re = kps[0], le = kps[1], nose = kps[2], mouth = kps[3];

    // Roll: angle of eye line
    const roll = Math.atan2(le.y - re.y, le.x - re.x) * 180 / Math.PI;

    // Yaw: nose position relative to eye span
    // Front camera is mirrored, so keypoints x are flipped.
    const eyeSpan = Math.hypot(le.x - re.x, le.y - re.y);
    const noseRatio = eyeSpan > 1e-4 ? (nose.x - re.x) / (le.x - re.x) : 0.5;
    const yaw = (noseRatio - 0.5) * 90;

    // Pitch: nose vertical position relative to eye-mouth span
    const midX = (re.x + le.x) / 2;
    const midY = (re.y + le.y) / 2;
    const vertSpan = Math.hypot(mouth.x - midX, mouth.y - midY);
    const noseFrac = vertSpan > 1e-4 ? ((nose.y - midY) / (mouth.y - midY)) : 0.5;
    const pitch = (noseFrac - 0.4) * -90;

    // Face normal vector (unit vector in 3D from yaw & pitch)
    const yawRad = yaw * Math.PI / 180;
    const pitchRad = pitch * Math.PI / 180;
    const nx = Math.sin(yawRad);
    const ny = -Math.sin(pitchRad);
    const nz = Math.cos(yawRad) * Math.cos(pitchRad);

    return { yaw, pitch, roll, normal: { x: nx, y: ny, z: nz } };
}
