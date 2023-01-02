import numpy as np
import cv2 as cv
import mediapipe as mp

def color_points(points_list):
    for i in points_list:
        # Acquire x, y but don't forget to convert to integer.
        x = int(i.x * image.shape[1])
        y = int(i.y * image.shape[0])
        # Annotate landmarks or do whatever you want.
        cv.circle(image, (x, y), 5, (0, 255, 0), -1)

def find_angle(landmark_list):
    """
    Finds angles of landmarks 11, 13, 15 and 12, 14, 16
    :param landmark_list:
    :type landmark_list:
    :return:
    :rtype:
    """
    angle = 0
    l_index = [11, 13, 15]
    r_index = [12, 14, 16]
    try:
        l_points = [e for i, e in enumerate(
            landmark_list) if i in l_index]
        print(l_points)
        r_points = [e for i, e in enumerate(
            landmark_list) if i in r_index]
        print(l_points)
        color_points(l_points)
        color_points(r_points)
        return l_points, r_points, angle
    except:
        landmarks = []
        print('hello world')
        return landmarks, 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # SET UP PACKAGE CALLS
    cap = cv.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)

    # Write some Text
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2
    count = 0

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5, enable_segmentation=False) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            image = cv.flip(image, 1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # print(results.pose_landmarks.landmark[0])

            # 11, 13, 15
            l_points, r_points, angle = find_angle(
                results.pose_landmarks.landmark)

            # Flip the image horizontally for a selfie-view display.
            cv.imshow('MediaPipe Pose', cv.flip(image, 1))
            if cv.waitKey(1) == ord('q'):
                # print(results.pose_landmarks.landmark[0])  # way to extract just the
                # specific landmark of an area
                break
            cv.putText(image, f"count: {str(int(count))}",
                       (100, 300),
                       font,
                       fontScale,
                       fontColor,
                       thickness,
                       lineType)
            cv.imshow("vid", image)
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
