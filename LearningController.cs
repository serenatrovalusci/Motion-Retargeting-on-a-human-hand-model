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

        private TcpClient client;
        private NetworkStream stream;
        private Thread socketThread;
        private float[] closureAbductionInput = new float[4]; // [thumbC, indexC, middleC, abduction]
        private float[] receivedJointAngles = new float[27];  // 9 joints × 3 Euler angles
        private bool newDataAvailable = false;
        private bool isRunning = true;

        void Start()
        {
            // Start background thread for socket
            socketThread = new Thread(SocketLoop);
            socketThread.IsBackground = true;
            socketThread.Start();
        }

        void Update()
        {
            //  Real-time input from WeArt thimbles
            if (thumbThimble != null && indexThimble != null && middleThimble != null)
            {
                closureAbductionInput[0] = thumbThimble.Closure.Value;
                closureAbductionInput[1] = indexThimble.Closure.Value;
                closureAbductionInput[2] = middleThimble.Closure.Value;
                closureAbductionInput[3] = thumbThimble.Abduction.Value;
            }

            //  Apply predicted joint rotations
            if (newDataAvailable)
            {
                for (int i = 0; i < jointTransforms.Length; i++)
                {
                    if (jointTransforms[i] != null)
                    {
                        int index = i * 3;
                        Vector3 predictedEuler = new Vector3(
                            receivedJointAngles[index],
                            receivedJointAngles[index + 1],
                            receivedJointAngles[index + 2]
                        );

                        jointTransforms[i].rotation = palmTransform.rotation * Quaternion.Euler(predictedEuler);
                    }
                }
                newDataAvailable = false;
            }
        }

        void SocketLoop()
        {
            try
            {
                client = new TcpClient(serverIP, serverPort);
                stream = client.GetStream();
                Debug.Log(" Connected to Python server.");

                byte[] inputBuffer = new byte[4 * 4];   // 4 floats
                byte[] outputBuffer = new byte[27 * 4]; // 27 floats

                while (isRunning)
                {
                    // Prepare input: 4 floats → byte array
                    Buffer.BlockCopy(closureAbductionInput, 0, inputBuffer, 0, inputBuffer.Length);
                    stream.Write(inputBuffer, 0, inputBuffer.Length);

                    // Read full 108 bytes (27 floats)
                    int totalRead = 0;
                    while (totalRead < outputBuffer.Length)
                    {
                        int bytesRead = stream.Read(outputBuffer, totalRead, outputBuffer.Length - totalRead);
                        if (bytesRead == 0) throw new Exception("Disconnected from server.");
                        totalRead += bytesRead;
                    }

                    // Convert to float array
                    float[] predictions = new float[27];
                    Buffer.BlockCopy(outputBuffer, 0, predictions, 0, outputBuffer.Length);
                    receivedJointAngles = predictions;
                    newDataAvailable = true;
                }
            }
            catch (Exception e)
            {
                Debug.LogError(" Socket error: " + e.Message);
            }
        }

        void OnApplicationQuit()
        {
            isRunning = false;
            stream?.Close();
            client?.Close();
            socketThread?.Abort();
        }
    }
}
