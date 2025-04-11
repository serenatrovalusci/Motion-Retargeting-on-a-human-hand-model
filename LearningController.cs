using System;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

namespace WeArt.Components
{
    public class LearningController : MonoBehaviour
    {
        [Header("Neural Hand Control")]
        [Tooltip("Ordered: thumb1, thumb2, thumb3, index1, index2, index3, middle1, middle2, middle3")]
        public Transform[] jointTransforms; // Assign in Inspector (should be 9)
        public Transform palmTransform;     // Palm Transform for relative rotation

        [Header("WeArt Thimbles")]
        [SerializeField] private WeArtThimbleTrackingObject thumbThimble;
        [SerializeField] private WeArtThimbleTrackingObject indexThimble;
        [SerializeField] private WeArtThimbleTrackingObject middleThimble;

        [Header("Socket Settings")]
        public string serverIP = "127.0.0.1";
        public int serverPort = 65432;

        [Header("Joint Constraints")]
        public float maxAngleChangePerFrame = 30f;
        public float thumbFlexionRange = 80f;
        public float fingerFlexionRange = 90f;
        public float abductionRange = 30f;

        private TcpClient client;
        private NetworkStream stream;
        private Thread socketThread;
        private float[] closureAbductionInput = new float[4];
        private float[] receivedJointAngles = new float[27];
        private Quaternion[] previousJointRotations = new Quaternion[9];
        private bool newDataAvailable = false;
        private bool isRunning = true;
        private object dataLock = new object();

        void Start()
        {
            // Initialize previous rotations
            for (int i = 0; i < jointTransforms.Length; i++)
            {
                if (jointTransforms[i] != null)
                {
                    previousJointRotations[i] = jointTransforms[i].localRotation;
                }
            }

            socketThread = new Thread(SocketLoop);
            socketThread.IsBackground = true;
            socketThread.Start();
        }

        void Update()
        {
            // Get real-time input from WeArt thimbles
            if (thumbThimble != null && indexThimble != null && middleThimble != null)
            {
                closureAbductionInput[0] = thumbThimble.Closure.Value;
                closureAbductionInput[1] = indexThimble.Closure.Value;
                closureAbductionInput[2] = middleThimble.Closure.Value;
                closureAbductionInput[3] = thumbThimble.Abduction.Value;
            }

            // Apply predicted joint rotations with thread-safe access
            if (newDataAvailable)
            {
                lock (dataLock)
                {
                    for (int i = 0; i < jointTransforms.Length; i++)
                    {
                        if (jointTransforms[i] == null) continue;

                        int index = i * 3;
                        Vector3 predictedEuler = new Vector3(
                            receivedJointAngles[index],
                            receivedJointAngles[index + 1],
                            receivedJointAngles[index + 2]
                        );

                        // Create target rotation in palm space
                        Quaternion targetRotation = Quaternion.Euler(predictedEuler);
                        
                        // Apply joint-specific constraints
                        targetRotation = ApplyJointConstraints(i, targetRotation);
                        
                        // Smooth rotation changes
                        Quaternion smoothedRotation = SmoothRotation(previousJointRotations[i], targetRotation);
                        
                        // Apply final rotation relative to palm
                        jointTransforms[i].rotation = palmTransform.rotation * smoothedRotation;
                        previousJointRotations[i] = smoothedRotation;
                    }
                    newDataAvailable = false;
                }
            }
        }

        private Quaternion SmoothRotation(Quaternion from, Quaternion to)
        {
            float angle = Quaternion.Angle(from, to);
            if (angle > maxAngleChangePerFrame)
            {
                float t = maxAngleChangePerFrame / angle;
                return Quaternion.Slerp(from, to, t);
            }
            return to;
        }

        private Quaternion ApplyJointConstraints(int jointIndex, Quaternion rotation)
        {
            Vector3 euler = rotation.eulerAngles;
            
            // Normalize angles to -180 to 180 range
            euler.x = NormalizeAngle(euler.x);
            euler.y = NormalizeAngle(euler.y);
            euler.z = NormalizeAngle(euler.z);

            // Apply joint-specific constraints
            switch (jointIndex)
            {
                // Thumb joints
                case 0: // thumb1 (CMC)
                    euler.x = Mathf.Clamp(euler.x, -thumbFlexionRange, thumbFlexionRange);
                    euler.y = Mathf.Clamp(euler.y, -abductionRange, abductionRange);
                    euler.z = Mathf.Clamp(euler.z, -30f, 30f);
                    break;
                    
                case 1: // thumb2 (MCP)
                case 2: // thumb3 (IP)
                    euler.x = Mathf.Clamp(euler.x, 0f, thumbFlexionRange);
                    euler.y = Mathf.Clamp(euler.y, -15f, 15f);
                    euler.z = Mathf.Clamp(euler.z, -15f, 15f);
                    break;
                    
                // Finger joints (index and middle)
                case 3: // index1 (MCP)
                case 6: // middle1 (MCP)
                    euler.x = Mathf.Clamp(euler.x, -20f, 20f);
                    euler.y = Mathf.Clamp(euler.y, -abductionRange, abductionRange);
                    euler.z = Mathf.Clamp(euler.z, -fingerFlexionRange, 0f);
                    break;
                    
                case 4: // index2 (PIP)
                case 5: // index3 (DIP)
                case 7: // middle2 (PIP)
                case 8: // middle3 (DIP)
                    euler.x = Mathf.Clamp(euler.x, -10f, 10f);
                    euler.y = Mathf.Clamp(euler.y, -10f, 10f);
                    euler.z = Mathf.Clamp(euler.z, -fingerFlexionRange, 0f);
                    break;
            }
            
            return Quaternion.Euler(euler);
        }

        private float NormalizeAngle(float angle)
        {
            while (angle > 180f) angle -= 360f;
            while (angle < -180f) angle += 360f;
            return angle;
        }

        void SocketLoop()
        {
            try
            {
                client = new TcpClient(serverIP, serverPort);
                stream = client.GetStream();
                Debug.Log("Connected to Python server.");

                byte[] inputBuffer = new byte[4 * 4];
                byte[] outputBuffer = new byte[27 * 4];

                while (isRunning)
                {
                    // Send input
                    Buffer.BlockCopy(closureAbductionInput, 0, inputBuffer, 0, inputBuffer.Length);
                    stream.Write(inputBuffer, 0, inputBuffer.Length);

                    // Receive output
                    int totalRead = 0;
                    while (totalRead < outputBuffer.Length && isRunning)
                    {
                        int bytesRead = stream.Read(outputBuffer, totalRead, outputBuffer.Length - totalRead);
                        if (bytesRead == 0) throw new Exception("Server disconnected");
                        totalRead += bytesRead;
                    }

                    // Update received data with thread safety
                    lock (dataLock)
                    {
                        Buffer.BlockCopy(outputBuffer, 0, receivedJointAngles, 0, outputBuffer.Length);
                        newDataAvailable = true;
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError("Socket error: " + e.Message);
            }
            finally
            {
                stream?.Close();
                client?.Close();
            }
        }

        void OnApplicationQuit()
        {
            isRunning = false;
            stream?.Close();
            client?.Close();
            
            if (socketThread != null && socketThread.IsAlive)
            {
                socketThread.Join(100); // Give 100ms to clean up
                if (socketThread.IsAlive)
                    socketThread.Abort();
            }
        }
    }
}
